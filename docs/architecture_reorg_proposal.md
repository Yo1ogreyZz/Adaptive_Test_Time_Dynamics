# 面向多任务/多模态的 ATTD + Mamba 通用框架重构建议

> 目标：把当前“模型实现 + 任务脚本”耦合较高的结构，升级为“**通用核心（Core）+ 可插拔任务（Task）+ 可插拔模态（Modality）+ 可插拔训练流程（Runtime）**”的分层架构。

## 1. 先定边界：什么是“通用框架”，什么是“任务特化”

建议你把仓库中的代码按责任切分成 4 层：

1. **Core（稳定层）**
   - Mamba backbone 抽象
   - Adapter 注入机制（LoRA/低秩/Prefix/Delta-SSM）
   - Controller（如 monotonic depth policy）
   - Test-time adaptation inner loop（优化器、更新步数、稳定性约束）
   - 统一的 `ModelOutput` / `Batch` 数据结构

2. **Modality（模态层）**
   - 文本编码（byte-level, tokenizer）
   - 图像 patch/pixel tokenizer
   - 序列化器（把任意输入变成 `(B, L, D)`）
   - mask/padding 规范

3. **Task（任务层）**
   - 文本分类、文本预测（LM）、Pathfinder、ListOps 等
   - 只定义：`loss_fn`、`metrics`、`label_schema`、`eval hooks`
   - 绝不直接依赖某个具体 backbone 细节

4. **Runtime（运行层）**
   - 训练/验证/测试流程
   - test-time adaptation 策略开关（none/fixed/adaptive）
   - 配置管理、checkpoint、日志

这样做的核心收益：
- 当你加新任务时，只写 Task +（必要时）Modality；Core 基本不动。
- 当你替换 backbone（Mamba/Transformer/Hybrid）时，Task 基本不动。
- 当你试验新 controller 或新 adapter 时，不会侵入任务代码。

---

## 2. 推荐目录结构（可渐进迁移）

```text
adaptive_dynamics/
  core/
    backbones/
      base.py
      mamba.py
      transformer.py (可选)
    adapters/
      base.py
      low_rank.py
      delta_ssm.py
    controllers/
      base.py
      monotonic.py
    tta/
      inner_loop.py
      objectives.py
      stability.py
    interfaces/
      batch.py
      outputs.py

  modalities/
    text/
      encoder.py
      collator.py
    vision/
      encoder.py
      collator.py
    sequence/
      common.py

  tasks/
    base.py
    text_classification/
      task.py
      metrics.py
    language_modeling/
      task.py
      metrics.py
    pathfinder/
      task.py
      metrics.py
    listops/
      task.py
      metrics.py

  runtime/
    trainer.py
    evaluator.py
    checkpoint.py
    registry.py

  configs/
    model/
    task/
    runtime/
    experiment/

  scripts/
    train.py
    eval.py
    sweep.py
```

---

## 3. 统一接口（这是重构成败关键）

### 3.1 Backbone 接口

```python
class Backbone(nn.Module):
    def forward(self, x, mask=None, adapter_state=None, return_features=False):
        ...
```

- 输入输出始终围绕 `(B, L, D)`，避免任务代码对内部结构有假设。
- adapter_state 独立传入，便于 test-time inner loop 改参数但不污染 backbone 本体。

### 3.2 Task 接口

```python
class Task:
    def preprocess(self, raw_batch) -> Batch: ...
    def forward(self, model, batch: Batch) -> ModelOutput: ...
    def compute_loss(self, output: ModelOutput, batch: Batch): ...
    def compute_metrics(self, output: ModelOutput, batch: Batch): ...
```

- 任务只关心标签和指标，不关心 adapter/controller 细节。

### 3.3 TTA 策略接口

```python
class TTAStrategy:
    def adapt(self, model, batch, objective) -> AdapterState: ...
```

可实现：
- `NoAdaptation`
- `FixedStepsAdaptation(k=1/3/5)`
- `AdaptiveStepsAdaptation(controller=...)`

---

## 4. 把 “Mamba + Adapter + Controller” 变成组合，而不是硬编码

你现在的思路里，`ATTDWrapper` 功能很多（backbone、adapter、controller、inner-loop 都在一起）。建议拆成：

- `BackboneModel`：纯表征，不做 test-time 更新。
- `AdapterManager`：管理哪些参数可更新、怎么 reset、如何导出 delta。
- `Controller`：给 `difficulty -> k`。
- `TTAEngine`：执行 inner-loop 更新。
- `TaskHead`：分类/LM/二分类头。

主流程由 runtime 组装：

```python
features = backbone(x, adapter_state=state)
logits = head(features)
if use_tta:
    state = tta_engine.adapt(...)
    features = backbone(x, adapter_state=state)
    logits = head(features)
```

这样你未来换成 LLM 或图像任务，只要换 head/encoder/task，而不是改 ATTD 核心。

---

## 5. 面向多模态的输入统一策略

你的任务横跨文本、图像、Pathfinder，本质都可以被序列化：

- 文本：token 序列
- 图像：patch 序列（或 pixel 序列）
- Pathfinder：网格展平序列

建议明确一条“**统一序列化契约**”：

```text
raw input -> modality encoder -> (tokens, mask, metadata) -> backbone
```

并且把 pooling 方式也做成策略：
- `MeanPool`
- `CLSLast`
- `MaskedMean`
- `TaskSpecificPooling`

避免每个任务里写一套 pooling 逻辑。

---

## 6. 配置系统建议：分层配置 + 注册表

你后面会有大量 ablation（不同 adapter/controller/objective），建议尽快上：

- **分层配置**：`model.yaml + task.yaml + runtime.yaml + experiment.yaml`
- **registry**：通过字符串实例化模块（如 `adapter=low_rank`, `controller=monotonic`）

这样你不需要在 `train.py` 里不断加 if-else，实验管理会清晰很多。

---

## 7. 训练流程建议：把“算法”和“工程”解耦

建议 Trainer 里明确区分两个 step：

1. `outer_step`：常规监督训练（更新 backbone/head/controller）
2. `tta_step`：测试时或模拟测试时的 inner adaptation

并支持三种 regime：
- 纯训练（不做 TTA）
- 训练时模拟 TTA（meta-style）
- 评估时 TTA（真实推理）

这样你可以系统比较：
- 不同 inner loss（entropy/consistency/LM）
- 不同 k policy（fixed/adaptive）
- 不同可训练参数子集（仅 adapter / adapter+norm）

---

## 8. 你这个项目最值得提前抽象的 5 个点

1. **`Batch` 数据结构统一**（inputs, labels, mask, aux）
2. **`ModelOutput` 统一**（logits, hidden, adapter_stats, controller_stats）
3. **inner objective 插件化**（分类熵、LM loss、重建损失）
4. **adapter 参数选择器**（哪些参数可做 test-time 更新）
5. **controller 输出协议**（hard k / soft k / cost term）

这 5 个点统一后，多任务扩展成本会明显下降。

---

## 9. 渐进式迁移路线（建议 3 个阶段）

### 阶段 A（1-2 天）——“先不动算法，先解耦文件结构”
- 把 task-specific model 从任务文件抽离到 `tasks/*/task.py`
- 引入 `Task` 基类
- `train.py` 改为通过 registry 创建 task/model

### 阶段 B（2-4 天）——“拆 ATTDWrapper”
- 把 adapter/controller/inner-loop 分拆成独立组件
- 增加 `TTAStrategy` 接口
- 保持当前实验结果可复现（写回归脚本）

### 阶段 C（持续）——“多模态规范化”
- 引入 vision/text modality encoder
- 跑通 text cls + pathfinder 同一 trainer
- 再接 text prediction（LM）

---

## 10. 给你当前仓库的一些针对性提醒

- 当前 `experiments/train.py` 同时承担参数解析、任务创建、模型拼装、恢复训练等多个职责，后续会成为扩展瓶颈。
- `tasks/*.py` 里既有数据逻辑又有模型头逻辑，建议拆分成 data/task/model_head。
- `ATTDWrapper` 现在聚合过多策略性逻辑，后续加 LLM 任务或视觉任务时会快速复杂化。

换句话说：**你现在最需要的是“接口稳定”，而不是“功能继续堆叠”。**

---

## 11. 一句落地建议（优先级最高）

如果你这周只能做一件事：

> 先定义并落地 `Task`、`Backbone`、`TTAStrategy` 三个接口，然后把现有 text/listops/pathfinder 接到这三个接口上。

一旦这一步完成，Mamba 和 Adapter 才真正变成“可复用框架”，而不是“某个实验脚本里的实现细节”。
