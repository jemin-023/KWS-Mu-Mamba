# Mathematical Foundations of KWS-Mu-Mamba

This document outlines the mathematical formulation of **$\mu$-Mamba** (Micro-Mamba with C1 Recurrence) and details the algorithmic optimizations that achieved a **14.0× speedup** and **39× memory reduction** over conventional embedded Mamba baselines on microcontrollers.

---

## 1. The Bottleneck in Standard Mamba (S6)

The continuous-time Selective State Space Model (SSM) maps an input signal $u(t) \in \mathbb{R}^{D}$ to an output $y(t) \in \mathbb{R}^{D}$ via an $N$-dimensional latent state $h(t) \in \mathbb{R}^{D \times N}$:

$$h'(t) = A h(t) + B(t) u(t)$$
$$y(t) = C(t) h(t) + D u(t)$$

Discretization with a zero-order hold (ZOH) using step size $\Delta_t$ yields:

$$\bar{A}_t = \exp(\Delta_t A), \qquad \bar{B}_t \approx \Delta_t B_t$$
$$h_t = \bar{A}_t \odot h_{t-1} + \bar{B}_t \otimes u_t$$
$$y_t = h_t C_t + D u_t$$

### Why Standard Mamba Fails on Resource-Constrained MCUs:
In the standard Mamba formulation:
1. $A \in \mathbb{R}^{D \times N}$ is learned as $A = -\exp(A_{\text{log}})$.
2. $\Delta_t \in \mathbb{R}^D$ is input-dependent: $\Delta_t = \text{softplus}(\text{dt\_proj}(u_t))$.
3. Consequently, the transition matrix must be computed **online at every time step $t$**:

$$\bar{A}_t[d, n] = \exp\Big(-\Delta_t[d] \cdot \exp(A_{\text{log}}[d, n])\Big)$$

For typical parameters ($D=128, N=16, T=100\text{ frames}$), evaluating $\bar{A}_t$ requires:

$$100 \times 128 \times 16 = \mathbf{204{,}800 \text{ calls to runtime } \texttt{expf()}}$$

On microcontrollers like the **ESP32-C3 (RISC-V RV32IMC)**, which lack a hardware floating-point unit (FPU) and vector transcendental accelerators, each software-emulated `expf()` call consumes **~2,000 clock cycles**, requiring **over 400 million cycles** just for discretization.

---

## 2. The C1 Recurrence Constraint

To eliminate runtime matrix exponentials without sacrificing modeling capacity, $\mu$-Mamba introduces the **C1 harmonic constraint**.

### 2.1 Fixed Harmonic State Transition Matrix
Instead of learning an arbitrary continuous matrix $A$, we fix $A$ to an unlearned, structured harmonic progression:

$$A[d, n] = -(n + 1), \qquad \text{for } n = 0, 1, \dots, N-1$$

Under this structure, discretization evaluates to:

$$\bar{A}_t[d, n] = \exp\big(\Delta_t[d] \cdot A[d, n]\big) = \exp\big(-\Delta_t[d] \cdot (n + 1)\big) = \Big[\exp(-\Delta_t[d])\Big]^{n + 1}$$

### 2.2 Dual Sigmoid-Softplus Parameterization
Let $p_t = \text{dt\_proj}(u_t) \in \mathbb{R}^{D}$ be the linear projection of the token. We define the step size $\Delta_t$ and the base decay scalar $q_t$ via coupled logistic functions:

$$\Delta_t = \text{softplus}(p_t) = \ln(1 + e^{p_t})$$
$$q_t = \text{sigmoid}(-p_t) = \frac{1}{1 + e^{p_t}}$$

Applying the fundamental algebraic identity:

$$\exp(-\Delta_t) = \exp\Big(-\ln(1 + e^{p_t})\Big) = \frac{1}{1 + e^{p_t}} = q_t$$

We arrive at the exact closed-form relation:

$$\bar{A}_t[d, n] = q_t[d]^{n + 1}$$

### 2.3 Successive Multiplication (Zero Transcendental Functions)
Because $\bar{A}_t[d, n] = q^{n+1}$, all $N$ state decay coefficients across the entire latent dimension derive from a **single scalar** $q_t[d] \in (0, 1)$:

$$P_0 = q, \qquad P_1 = P_0 \cdot q = q^2, \qquad \dots, \qquad P_{N-1} = P_{N-2} \cdot q = q^N$$

$$\bar{A}_t[d] = \begin{bmatrix} q & q^2 & q^3 & \dots & q^N \end{bmatrix}$$

$$\boxed{\text{Runtime Transcendental Exponentials: } D \times N \longrightarrow \mathbf{0}}$$

The entire discretization step reduces to **1 lookup** for $q = \text{sigmoid}(-p)$ followed by **$N-1$ integer multiplications**.

---

## 3. Streaming Mathematical Formulation

The complete per-token recurrence at step $t$ executes as follows:

1. **RMSNorm & Projection**:
   $$x_t = \text{RMSNorm}(token_t)$$
   $$\begin{bmatrix} u_t \\ z_t \end{bmatrix} = W_{\text{in}} x_t \qquad (u_t, z_t \in \mathbb{R}^{D_{\text{inner}}})$$

2. **Causal Conv1d via Ring Buffer**:
   $$u_{\text{conv}, t} = W_{\text{conv}} \star [u_{t-K+1}, \dots, u_t] + b_{\text{conv}}$$
   $$u'_t = \text{SiLU}(u_{\text{conv}, t})$$

3. **SSM Parameter Projections**:
   $$\begin{bmatrix} p_t \\ B_t \\ C_t \end{bmatrix} = W_x u'_t \qquad (p_t \in \mathbb{R}^R, \; B_t, C_t \in \mathbb{R}^N)$$
   $$p_t = W_{\Delta} p_t + b_{\Delta} \qquad (p_t \in \mathbb{R}^{D_{\text{inner}}})$$

4. **C1 Recurrence Update**:
   $$q_t = \text{sigmoid}(-p_t), \qquad \Delta_t = \text{softplus}(p_t)$$
   $$v_t = \Delta_t \odot u'_t$$
   $$dB_t = v_t \otimes B_t \in \mathbb{R}^{D_{\text{inner}} \times N}$$
   $$P_t[d, n] = q_t[d]^{n+1}$$
   $$\mathbf{h}_t = P_t \odot \mathbf{h}_{t-1} + dB_t$$

5. **Output Gating**:
   $$y_t = \mathbf{h}_t C_t + D_{\text{skip}} \odot u'_t$$
   $$y_{\text{gate}, t} = y_t \odot \text{SiLU}(z_t)$$
   $$\text{out}_t = token_t + W_{\text{out}} y_{\text{gate}, t}$$

---

## 4. W8A16 Fixed-Point Quantization & Look-Up Tables

To run on integer-only RISC-V cores without FPU instructions, all operations are mapped to signed fixed-point formats.

### 4.1 Symmetric Quantization with Multiplier-Shift Scaling
Weights are quantized to symmetric 8-bit integers:

$$W_{\text{int8}} = \text{clamp}\left(\left\lfloor \frac{W_{\text{float}}}{S_w} \right\rceil, -127, 127\right)$$

During matrix multiplications, the dot product is accumulated in 32-bit registers:

$$\text{Acc} = \sum_{i} W_{\text{int8}, i} \cdot X_{\text{int16}, i}$$

The result is rescaled to the output fixed-point fractional format using an integer multiplier $M \in [1, 32767]$ and right shift $S \in [0, 31]$:

$$Y_{\text{int16}} = \text{clamp}\left( \frac{\text{Acc} \times M + 2^{S-1}}{2^S}, -32768, 32767 \right)$$

### 4.2 Look-Up Tables (LUTs) for Non-Linearities
Instead of computing non-linearities iteratively, $\text{sigmoid}$ and $\text{SiLU}$ are precomputed into 16-bit Look-Up Tables:
- **`mm_lut_sigmoid_q15`**: $1024 \times 2\text{ bytes} = 2.0\text{ KB}$
- **`mm_lut_silu_q15`**: $2560 \times 2\text{ bytes} = 5.1\text{ KB}$
- **Total LUT footprint**: **$7.1\text{ KB}$** in flash memory.

At runtime, resolving $q = \text{sigmoid}(-p)$ requires only a clamping operation and a direct memory index:

$$q = \text{LUT}_{\text{sigmoid}}[\text{index}(p)] \quad \Longrightarrow \quad \mathcal{O}(1) \text{ single-cycle lookup}$$

---

## 5. Streaming Memory Topology: $\mathcal{O}(1)$ vs $\mathcal{O}(T)$

| Memory Dimension | Offline Batch (MambaLite-Micro) | Streaming $\mu$-Mamba (Ours) |
| :--- | :--- | :--- |
| **Input Buffer** | Full audio spectrogram: $T \times F = 100 \times 40 = 4{,}000\text{ floats}$ | Single audio frame: $512\text{ PCM samples} = 1{,}024\text{ bytes}$ |
| **Intermediate Activations** | Tensors allocated along time dimension $T=100$:<br>• $xz: 256 \times 100 \times 4\text{ B} = 102.4\text{ KB}$<br>• $x, z: 2 \times 128 \times 100 \times 4\text{ B} = 102.4\text{ KB}$<br>• $y_{\text{out}}: 128 \times 100 \times 4\text{ B} = 51.2\text{ KB}$ | **Zero intermediate sequence allocation**.<br>Processed strictly token-by-token ($T=1$). |
| **Recurrent Persistent State** | None (Recalculated over the window) | • Conv1d ring buffer: $(K-1) \times D_{\text{inner}} \times 2\text{ L} = \mathbf{768\text{ B}}$<br>• Latent state $h$: $D_{\text{inner}} \times N \times 2\text{ L} \times 4\text{ B} = \mathbf{4{,}096\text{ B}}$ |
| **Peak Runtime RAM** | **$234.36\text{ KB}$** (Exhausts SRAM, fragmentation) | **$< 6.0\text{ KB}$** (Static BSS context, $0\text{ malloc}$) |

---

## 6. Summary: What Made the 14.0× Speedup Possible

1. **Elimination of Transcendental Functions ($204{,}800 \rightarrow 0$)**: The C1 harmonic constraint turns matrix exponentiation into scalar integer multiplications.
2. **Fixed-Point Arithmetic (No Software Float Emulation)**: W8A16 integer operations execute natively on the single-cycle RV32IMC integer ALU, avoiding multi-cycle floating-point library routines.
3. **$\mathcal{O}(1)$ Constant Memory Layout**: Streaming token-by-token inference keeps working buffers in L1 data cache/SRAM, avoiding the memory bandwidth bottlenecks of large multi-kilobyte tensor allocations.
4. **LUT-Accelerated Activations**: 16-bit precomputed tables replace polynomial approximations for $\text{sigmoid}$ and $\text{SiLU}$.
