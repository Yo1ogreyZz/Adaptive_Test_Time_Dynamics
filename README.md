# Adaptive Test-Time Dynamics (ATTD) on Mamba

This repository contains our DSA5204 project on **Adaptive Test-Time Dynamics (ATTD)**: a framework that combines **test-time adaptation** with **adaptive computation depth**, implemented on the **Mamba** architecture.

***

## 1. What this project is about

Most models, including Mamba, are used in a simple way at test time:

- The model parameters are frozen.
- Every input uses the same amount of compute.

This is wasteful and fragile when:

- The test distribution shifts away from training.
- Some inputs are clearly harder than others and would benefit from more “thinking”.

ATTD aims to let the model, at test time:

- Adjust a small, structured part of its **dynamics** (instead of all weights).
- Decide **how many** test-time update steps to run per input based on difficulty.

In short, we want to learn both:

- *What* to adapt (a small “dynamics adapter” inside Mamba).
- *How much* to adapt (input-dependent number of update steps).

***

## 2. Core ideas

### Structured dynamics adapter

Inside each Mamba block, there is a state transition module that controls how the hidden state evolves. Instead of changing the whole block at test time, we add a small, low-rank “adapter” on top of the original transition. 

During normal training, Mamba is trained as usual. At test time, the big backbone stays frozen, and only this small adapter is allowed to change. This gives us:

- A clear target for adaptation: the internal dynamics.
- A small parameter set, which is easier to control and more stable.

### Ponder signal from an internal loss

For a given test input, we compute an **internal self-supervised loss**, such as:

- A language modelling loss on the sequence, or  
- A reconstruction / consistency loss.

This scalar loss is used as a **difficulty signal** (ponder signal):

- If the internal loss is low, the current dynamics are already well matched to this input.  
- If it is high, the model is struggling and might benefit from extra adaptation steps.

### Monotonic controller for adaptation depth

We then feed this difficulty signal into a very simple controller that outputs an integer “depth”:

- Easy inputs → small depth (few or zero adaptation steps).  
- Hard inputs → larger depth (more steps).

The controller is **monotonic by design**: as the difficulty signal increases, the chosen depth never decreases. This matches the intuitive idea that harder inputs should never get less compute than easier ones.

### Test-time inner loop

Given the chosen depth:

- We start from an initial version of the dynamics adapter.
- We run a small number of gradient-based update steps on this adapter, using the internal loss as the objective.
- After these updates, we keep the backbone fixed, use the adapted dynamics to run Mamba on the input, and produce the final prediction.

Only the adapter is changed at test time; the backbone weights stay untouched.

***

## 3. Theoretical intuition

Our theoretical guide focuses on three questions:

1. **Stability**  
   Because we only change a low-rank adapter and take small update steps, the change in the internal dynamics is controlled. Together with suitable regularisation on the size of each update, this helps keep the model’s state evolution stable rather than exploding or diverging.

2. **Reasonable controller behaviour**  
   The controller is monotonic by construction: harder inputs (higher internal loss) always get at least as much compute as easier inputs. Under natural assumptions like “more adaptation cannot hurt too much” and “benefits of extra steps gradually diminish”, it can be argued that such monotone policies are close to optimal for allocating a fixed compute budget.

3. **Bi-level / meta-learning view**  
   Conceptually, there is an outer loop that trains the backbone, the initial adapter and the controller, and an inner loop that adapts the adapter at test time. This is similar to ideas from meta-learning and bilevel optimisation (such as MAML), where the outer training takes into account how the model will adapt later.

***

## 4. Relation to existing work

ATTD is inspired by and connects to several threads:

- **PonderTTT**: learns when to apply a single TTT update to an LLM, based on a self-supervised signal and a compute budget.  
- **Test-Time Training (TTT)**: performs self-supervised tuning at test time to handle distribution shift.  
- **Adaptive computation** (e.g. ACT, PonderNet): learns halting policies and expected depth penalties.  
- **Structured state space models** (Mamba, HiPPO, S4): provide a dynamics-focused view and tools to reason about stability and long-range behaviour.  
- **Meta-learning / bilevel methods**: give the formal language for “outer loop training, inner loop adaptation”.

Our twist is to:

- Make the test-time adaptation explicitly about **dynamics** (inside Mamba), rather than arbitrary weights.  
- Learn a controller that chooses a **variable number of adaptation steps per input**, rather than just “update or not”.

***

## 5. Benchmarks

We plan to evaluate ATTD-Mamba on **Long Range Arena (LRA)** tasks:

- **ListOps** – long-range symbolic/structural reasoning  
- **Pathfinder** – long-range visual reasoning  
- **Text** – long-context text modelling

For each task, we compare:

- Plain Mamba (no test-time adaptation)  
- Mamba with fixed-number TTT for all inputs  
- PonderTTT-style “ponder or not” gating  
- **ATTD-Mamba** with structured dynamics and adaptive depth

The main questions we want to answer are:

- Can ATTD achieve better robustness than fixed Mamba with similar average compute?  
- Does focusing adaptation on dynamics help stability and performance compared to generic TTT?
