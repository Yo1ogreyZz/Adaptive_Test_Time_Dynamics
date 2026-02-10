# Adaptive Test-Time Dynamics (ATTD) on Mamba

This repository contains our implementation of **Adaptive Test-Time Dynamics (ATTD)**: a framework that combines **test-time adaptation** with **adaptive computation depth**, instantiated on the **Mamba** architecture.

***

## 1. What We Are Trying to Do

Standard deep models (including Mamba) are usually used with:

- **Frozen parameters** at test time  
- **Fixed compute** per input (same depth / cost for all samples)

This is problematic when:

- The test distribution shifts away from training  
- Some inputs are much harder than others and would benefit from extra computation

Our goal is to let the model, at test time:

1. **Adapt its internal dynamics** (not all parameters, but a structured subset), and  
2. **Decide how many adaptation steps to take** per input based on difficulty.

In other words, ATTD tries to jointly learn:

- *What* to adapt (a low-rank “dynamics adapter” inside Mamba), and  
- *How much* to adapt (an input-dependent number of test-time update steps).

***

## 2. Model and Method (Conceptual)

We work with a Mamba-style **state space model**. Each Mamba block has a state transition matrix \(A_0\) that governs the dynamics.

### 2.1 Structured Dynamics Adapter

Instead of adapting all weights, we introduce a small, low-rank adapter:

\[
A_k = A_0 + U_k V_k^\top
\]

- \(A_0\): fixed base dynamics (learned during training and then frozen at test time)  
- \(U_k, V_k\): low-rank factors updated during test-time adaptation (this is our \(\psi_k\))

Only \((U_k, V_k)\) are updated at test time, making the adaptation:

- **Structured** (explicitly tied to dynamics)  
- **Lightweight** (few additional parameters)

### 2.2 Ponder Signal (Internal Loss)

Given a test input \(x\), we define an **internal self-supervised loss**:

\[
L_{\text{int}}(x; \theta_{\text{base}}, \psi_0)
\]

Examples:

- Language modelling loss on the input sequence  
- Reconstruction or consistency loss

We interpret the scalar value

\[
s = L_{\text{int}}(x; \theta_{\text{base}}, \psi_0)
\]

as a **ponder / difficulty signal**: higher \(s\) means the current dynamics do not fit this input well and may need more adaptation.

### 2.3 Adaptive Depth Controller

We use a simple, **monotonic, threshold-based controller**:

\[
K_{\text{dyn}} = C(s) = \sum_{i=1}^{K_{\max}} \mathbf{1}(s > t_i),
\quad t_1 < t_2 < \dots < t_{K_{\max}}
\]

- \(K_{\text{dyn}}\): number of test-time adaptation steps for this input  
- \(\{t_i\}\): learnable thresholds

This ensures:

- If an input is “easier” (smaller \(s\)), then \(K_{\text{dyn}}\) is smaller  
- If it is “harder” (larger \(s\)), then \(K_{\text{dyn}}\) is larger

This is the “how much to adapt” part.

### 2.4 Test-Time Inner Loop (Dynamics-Only Adaptation)

Starting from initial dynamics \(\psi_0 = (U_0, V_0)\), we run \(K_{\text{dyn}}\) gradient steps:

\[
\psi_{k+1} = \psi_k - \eta \nabla_{\psi_k} L_{\text{int}}(x; A_k), 
\quad A_k = A_0 + U_k V_k^\top
\]

- The base Mamba parameters \(\theta_{\text{base}}\) are **not** changed at test time  
- Only \((U_k, V_k)\) are updated according to the internal loss

After \(K_{\text{dyn}}\) steps, we use the final dynamics \(A_{K_{\text{dyn}}}\) to produce the task prediction.

***

## 3. Training Objective (Theory Basis)

During training (on held-out data or a proxy distribution), we optimise a **joint objective**:

\[
L_{\text{total}} = 
\mathbb{E}[L_{\text{task}}]
+ \lambda \,\mathbb{E}[\text{Cost}(K_{\text{dyn}})]
+ \beta \,\mathbb{E}\Big[\sum_{k<K_{\text{dyn}}}\|\Delta\psi_k\|^2\Big]
\]

- \(L_{\text{task}}\): main task loss (e.g. classification or sequence loss)  
- \(\text{Cost}(K_{\text{dyn}})\): penalises large \(K_{\text{dyn}}\) to keep average compute under control  
  - e.g. \(\text{Cost}(K_{\text{dyn}}) = K_{\text{dyn}}\) or a convex function of \(K_{\text{dyn}}\)  
- \(\sum\|\Delta\psi_k\|^2\): regularises parameter changes between inner steps, encouraging **stable** dynamics updates  
- \(\lambda, \beta\): hyperparameters controlling compute–performance–stability trade-offs

This objective connects to:

- **PonderTTT / adaptive computation**: cost term encourages economical use of extra compute  
- **TTT / meta-learning / bilevel optimisation**: differentiating through test-time updates  
- **SSM stability theory**: low-rank updates and bounded spectral radius for \(A_k\)

***

## 4. Theoretical Intuition

The theoretical guide we use focuses on three questions:

1. **Stability of dynamics**  
   - Under low-rank updates \(A_k = A_0 + U_k V_k^\top\) with small, bounded steps, we can bound how much the spectral radius \(\rho(A_k)\) deviates from \(\rho(A_0)\).  
   - With appropriate learning rate and regularisation, we expect \(\rho(A_k) < 1\) throughout inner-loop adaptation (no exploding dynamics).

2. **Monotonic controller as a principled policy**  
   - The multi-threshold form of \(C(s)\) is monotone by construction.  
   - Under mild assumptions (e.g. loss decreases with more steps but with diminishing returns), an optimal allocation of compute tends to be a monotone function of difficulty, which matches this design.

3. **Bilevel / meta-learning view**  
   - Outer loop: learns \(\theta_{\text{base}}\), thresholds \(\{t_i\}\), and possibly initial \(\psi_0\).  
   - Inner loop: adapts \(\psi\) via test-time updates.  
   - With appropriate smoothness assumptions, gradients through the inner loop are well-defined, connecting to MAML-style analysis.

***

## 5. Benchmarks and Intended Use

We plan to test ATTD-Mamba on **Long Range Arena (LRA)** tasks:

- **ListOps** – compositional reasoning on nested lists  
- **Pathfinder** – long-range visual path finding  
- **Text** – long-context text modelling

We will compare:

- Fixed Mamba (no adaptation)  
- Full TTT (same number of steps for all inputs)  
- PonderTTT-style “ponder or not” gating  
- **ATTD-Mamba** (structured dynamics + adaptive depth)