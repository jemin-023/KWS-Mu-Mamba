# Mathematical Foundations of KWS-Mu-Mamba

This document details the mathematical formulation of **$\mu$-Mamba** (Micro-Mamba with C1 Recurrence), provides rigorous error and numerical stability derivations, and presents an architectural attribution of the optimizations enabling real-time streaming keyword spotting on low-power microcontrollers (specifically the **ESP32-C3 RISC-V RV32IMC**).

---

## 1. The Discretization Bottleneck in Standard Mamba (S6)

The continuous-time Selective State Space Model (SSM) maps an input signal $u(t) \in \mathbb{R}^{D_{\text{inner}}}$ to an output $y(t) \in \mathbb{R}^{D_{\text{inner}}}$ through an $N$-dimensional latent state $h(t) \in \mathbb{R}^{D_{\text{inner}} \times N}$:

$$\frac{d}{dt} h(t) = A h(t) + B(t) u(t)$$
$$y(t) = C(t) h(t) + D_{\text{skip}} \odot u(t)$$

where $D_{\text{skip}} \in \mathbb{R}^{D_{\text{inner}}}$ is the learned direct feedthrough (skip) vector, and $D_{\text{inner}}$ denotes the expanded inner channel dimension.

Discretizing this system across a variable time step $\Delta_t$ via zero-order hold (ZOH) yields the discrete-time recurrence:

$$\bar{A}_t = \exp(\Delta_t A)$$
$$\bar{B}_t = (\Delta_t A)^{-1} \big(\exp(\Delta_t A) - I\big) \Delta_t B_t \approx \Delta_t B_t$$
$$h_t = \bar{A}_t \odot h_{t-1} + \bar{B}_t \otimes u_t$$
$$y_t = h_t C_t + D_{\text{skip}} \odot u_t$$

*(Note: $\bar{B}_t \approx \Delta_t B_t$ is the standard first-order Euler approximation adopted across empirical Mamba S6 implementations).*

### 1.1 The Computational Cost of Online Discretization
In the unconstrained Mamba formulation:
1. The continuous transition matrix $A \in \mathbb{R}^{D_{\text{inner}} \times N}$ is parameterized as $A = -\exp(A_{\text{log}})$ with learned parameters $A_{\text{log}} \in \mathbb{R}^{D_{\text{inner}} \times N}$.
2. The step size $\Delta_t \in \mathbb{R}^{D_{\text{inner}}}$ is input-dependent: $\Delta_t = \text{softplus}(\tilde{p}_t)$, where $\tilde{p}_t \in \mathbb{R}^{D_{\text{inner}}}$ is the raw delta projection.
3. Because both $\Delta_t$ and $A$ vary across channels and states, the discrete transition matrix must be computed **online at every time step $t$**:

$$\bar{A}_t[d, n] = \exp\Big(-\Delta_t[d] \cdot \exp(A_{\text{log}}[d, n])\Big)$$

For an offline-window embedded configuration such as **MambaLite-Micro** ($D_{\text{inner}} = 128, N = 16, L = 1, T = 100\text{ frames}$, dimensions from its published weights header):
$$\text{Total } \texttt{expf()} \text{ evaluations per clip} = T \times D_{\text{inner}} \times N = 100 \times 128 \times 16 = \mathbf{204{,}800}$$

On microcontrollers without a hardware floating-point unit (FPU), such as the **ESP32-C3 (RV32IMC)**, `expf()` is a software routine. We measured the newlib `expf` on the C3 at **2,831 cycles per call** (and a soft-float multiply-accumulate at 227 cycles) with a micro-benchmark (`paper/data/board_f32_final_boot.txt`, ESP-IDF `-O2`, build of 2026-10-30; the same values, 2,831 and 227, were obtained in an earlier build of 2026-10-11). Multiplying, $204{,}800 \times 2{,}831 \approx 5.8 \times 10^8$ cycles of `expf` alone per 1.0 s window (derived, not a measurement of MambaLite-Micro itself, which was run on the C3 in this work but did not give consistent measurements, see §6.3).

---

## 2. The C1 Recurrence Constraint

To eliminate runtime matrix exponentiations, $\mu$-Mamba introduces the **C1 harmonic recurrence constraint**.

### 2.1 Fixed Harmonic Transition Matrix
Rather than treating $A$ as an unconstrained learned matrix, we freeze $A$ to an unlearned harmonic progression based on the S4D-Real diagonal initialization (Gu et al., 2022):

$$A[d, n] = -(n + 1), \qquad \text{for } n \in \{0, 1, \dots, N-1\}$$

Under this structure, the continuous state decay rates are fixed to the harmonic sequence $\{-1, -2, -3, \dots, -N\}$. Discretizing with step size $\Delta_t[d]$ gives:

$$\bar{A}_t[d, n] = \exp\big(\Delta_t[d] \cdot A[d, n]\big) = \exp\big(-\Delta_t[d] \cdot (n + 1)\big) = \Big[\exp(-\Delta_t[d])\Big]^{n + 1}$$

### 2.2 Dual Sigmoid-Softplus Parameterization
Let $r_t \in \mathbb{R}^R$ be the low-rank projection of the token ($R \ll D_{\text{inner}}$), and let $\tilde{p}_t = W_{\Delta} r_t + b_{\Delta} \in \mathbb{R}^{D_{\text{inner}}}$ be the raw pre-activation delta vector. We couple the step size $\Delta_t$ and the base decay factor $q_t$ through the following dual definitions:

$$\Delta_t = \text{softplus}(\tilde{p}_t) = \ln(1 + e^{\tilde{p}_t})$$
$$q_t = \text{sigmoid}(-\tilde{p}_t) = \frac{1}{1 + e^{\tilde{p}_t}}$$

Applying the exponential identity:

$$\exp(-\Delta_t) = \exp\Big(-\ln(1 + e^{\tilde{p}_t})\Big) = \frac{1}{1 + e^{\tilde{p}_t}} = q_t$$

Substituting this back into the discretized transition formula yields the exact equality:

$$\boxed{\bar{A}_t[d, n] = q_t[d]^{n + 1}}$$

### 2.3 Successive Multiplication Property
Because $\bar{A}_t[d, n]$ is an exact power series of $q_t[d] \in (0, 1)$, all $N$ state decay coefficients across state dimension $n \in [0, N-1]$ are generated by **$N-1$ scalar multiplications**:

$$P_0 = q_t[d]$$
$$P_1 = P_0 \cdot q_t[d] = q_t[d]^2$$
$$\vdots$$
$$P_{N-1} = P_{N-2} \cdot q_t[d] = q_t[d]^N$$

$$\bar{A}_t[d] = \begin{bmatrix} P_0 & P_1 & P_2 & \dots & P_{N-1} \end{bmatrix} = \begin{bmatrix} q_t[d] & q_t[d]^2 & q_t[d]^3 & \dots & q_t[d]^N \end{bmatrix}$$

Evaluating $\bar{A}_t$ across all channels therefore requires only **1 base lookup** for $q_t = \text{sigmoid}(-\tilde{p}_t)$ followed by **$N-1$ integer multiplications**, reducing runtime transcendental matrix exponentiations from $D_{\text{inner}} \times N \rightarrow 0$.

### 2.4 Expressivity Boundaries & Theoretical Capacity Limits
Freezing $A[d, n] = -(n + 1)$ restricts the hypothesis class compared to unconstrained Mamba:
1. In unconstrained Mamba, $A \in \mathbb{R}^{D_{\text{inner}} \times N}$ provides $D_{\text{inner}} \times N$ learnable parameters, permitting channel-specific decay ratios and arbitrary pole placements along the negative real axis.
2. In C1 recurrence, the relative decay ratios across state dimensions are fixed to the harmonic progression $1 : 2 : 3 : \dots : N$.
3. All temporal adaptability is governed dynamically by the input-dependent projection $\tilde{p}_t$, which shifts the effective time constant $\tau_n(t) = \frac{1}{\Delta_t[d](n + 1)}$ uniformly across the harmonic ladder for channel $d$.
4. **Empirical performance and the learned-$A$ ablation**: on the full official test set (4,890 clips, 12 classes) the C1-constrained models reach **92.31 %** (clip-trained baseline) and **89.02 %** (streaming-trained paper model) top-1 accuracy with the fixed engine, 95 % Wilson intervals $[91.53, 93.02]$ and $[88.11, 89.86]$ (FP32 C engine: 92.37 % and 89.14 %) <!-- paper/data/eval_full_test_shipped.json, eval_full_test_final.json -->. The cost of freezing $A$ was measured by training otherwise identical models with a learned $A = -\exp(A_{\log})$ (initialised to $-(n+1)$, discretised with $\exp(\Delta A)$, same recipe), over two seeds: see §2.6 <!-- paper/data/ablation_c1_vs_learned_a.json -->.

### 2.5 Exact Continuous ZOH Formulation (Zero-Softplus Direction)
Under exact continuous Zero-Order Hold (ZOH) discretization (rather than the Euler approximation $\bar{B} \approx \Delta B$), the input matrix evaluates to:

$$\bar{B}_t = (\Delta_t A)^{-1} \big(\exp(\Delta_t A) - I\big) \Delta_t B_t = A^{-1} \big(\exp(\Delta_t A) - I\big) B_t$$

Notice that $\Delta_t$ cancels entirely. Substituting the C1 diagonal structure $A[d, n] = -(n + 1)$:

$$\bar{B}_t[d, n] = -\frac{1}{n + 1} \big(q_t[d]^{n+1} - 1\big) B_t[n] = \boxed{\frac{1 - P_n}{n + 1} B_t[n]}$$

This continuous ZOH formulation offers notable mathematical properties:
1. **Zero Softplus / No $\Delta$**: $\bar{B}_t[d, n]$ depends exclusively on $P_n = q_t[d]^{n+1}$, which is already computed for state decay. This completely eliminates the need for $\Delta_t$ and removes the softplus LUT.
2. **Bounded Saturation**: As $\Delta \to \infty$ ($q \to 0, P_n \to 0$), the discrete input gain saturates gracefully to $\frac{1}{n+1} B_t[n]$, preventing numerical blowup in the regime where the Euler approximation $\Delta_t B_t$ grows unbounded.
3. Implementing exact ZOH requires retraining with the $\frac{1 - P_n}{n+1}$ formulation and introduces one additional multiply per $(d, n)$ pair. **It is not implemented**: the shipped model and engine use the Euler input gain $\bar B \approx \Delta B$ (verified numerically: the closed form above matches the exact ZOH to a relative error $\le 6.1 \times 10^{-14}$, `tests/test_math_claims.py`).

### 2.6 Cost of freezing $A$ and of the C1 recurrence (measurements)
**Accuracy cost of freezing $A$ (measured, PyTorch only, three seeds).** Per seed, two models were trained from scratch with the identical recipe: frozen C1 $A[d,n] = -(n+1)$, and a learned $A = -\exp(A_{\log})$ initialised to the same values and discretised with $\exp(\Delta A)$ (+1,024 parameters, 19,084 total). Both use the from-scratch streaming-sequence recipe (multi-clip sequences, 60 epochs, 160 samples per class, batch 16), not the recipe of the paper model. Test clip accuracy on the 4,890 clips (zero-state 50-token, last-token protocol), 95 % Wilson interval <!-- paper/data/ablation_c1_vs_learned_a.json; seeds 42 and 1 only in ablation_c1_vs_learned_a_seeds42_1.json -->:

| Seed | Frozen C1 $A$ | Learned $A$ | Difference |
| :--- | :--- | :--- | ---: |
| 42 | 79.47 % [78.31, 80.58] | 83.03 % [81.95, 84.05] | +3.56 pt |
| 1 | 72.80 % [71.54, 74.03] | 78.75 % [77.58, 79.88] | +5.95 pt |
| 2 | 80.65 % [79.52, 81.74] | 82.52 % [81.43, 83.55] | +1.86 pt |
| mean $\pm$ sd ($n=3$) | $77.64 \pm 4.23$ | $81.43 \pm 2.33$ | $+3.79 \pm 2.05$ |

Learned $A$ was better in all three seeds, but a paired $t$-test over three seeds is not significant ($t = 3.19$, 2 degrees of freedom, $p = 0.086$), and the spread of the frozen model across seeds (7.9 points between best and worst) exceeds the mean difference of the arms. The direction is consistent; the size is imprecise; and the penalty under the paper model's recipe is unknown. The learned values stay close to the harmonic ladder on average (seed 42, mean $|A|$ for $n = 0..7$: 1.02, 1.91, 2.88, 3.76, 4.85, 5.69, 6.62, 7.77) but spread across channels. There is no C engine for learned $A$, so no fixed-point or on-device figure exists for it. AdamW weight decay ($10^{-4}$, the same as for every other parameter) was applied to $A_{\log}$ in all these runs; Mamba implementations usually exclude it, and no run without it was made, so its effect on the comparison is not measured (it should be small at this value). Seeds 42 and 1 were trained locally (PyTorch 2.14.1, RTX 4050) and seed 2 on a Tesla T4 (PyTorch 2.11.0), recorded per seed in the JSON; they are independent runs, not exact reproductions.

**Speed cost of the C1 trick.** Measured on the board in soft-float: a per-state $\exp(-\Delta(n+1))$ decay costs 1.26x the C1 step (+1.91 M cycles per token, §6.2).

---

## 3. Streaming Model Architecture & Dimensional Comparison

To avoid notation ambiguity, the model parameters and topological dimensions are summarized below:

| Dimension Symbol | Definition | Baseline (MambaLite-Micro) | Streaming $\mu$-Mamba (Ours) |
| :--- | :--- | :--- | :--- |
| $D_{\text{model}}$ | Model feature dimension | $64$ | $32$ |
| $D_{\text{inner}}$ | Expanded channel dimension | $128$ | $64$ |
| $N$ | SSM state dimension per channel | $16$ | $8$ |
| $K$ | Causal depthwise Conv1d kernel size | $4$ | $4$ |
| $R$ | Rank of delta projection ($r_t$) | $4$ | $4$ |
| $L$ | Number of Mamba layers | $1$ | $2$ |
| $C$ | Output classification classes | $10$ | $12$ |
| Parameters | Total model parameter count | $\mathbf{35{,}914}$ | $\mathbf{18{,}060}$ |

*(Baseline parameter verification: `mambakws-any-10/include/mamba_weights.h` contains: `linear_in` = $40 \times 64 + 64 = 2{,}624$; Mamba block = In-proj $16{,}384$ + Conv $512 + 128$ + X-proj $4{,}608$ + Dt-proj $512 + 128$ + A-log $2{,}048$ + D $128$ + Out-proj $8{,}192 = 32{,}640$; `classifier` = $64 \times 10 + 10 = 650$; Total = **$35{,}914$ parameters**).*

### 3.1 Per-Token Streaming Step Algorithm
At each time step $t$ (representing a 20 ms hop of 320 PCM audio samples), the streaming inference engine executes sequentially without sequence buffering:

1. **Integer RMS Normalization**:
   $$x_t = \frac{\text{token}_t}{\sqrt{\frac{1}{D_{\text{model}}} \sum_{i} \text{token}_{t, i}^2 + \epsilon}} \odot w_{\text{norm}}$$

2. **Input Projection**:
   $$\begin{bmatrix} u_t \\ z_t \end{bmatrix} = W_{\text{in}} x_t, \qquad u_t, z_t \in \mathbb{R}^{D_{\text{inner}}}$$

3. **Causal Depthwise Conv1d (Ring Buffer)**:
   A ring buffer maintains the $K - 1 = 3$ prior activations:
   $$u_{\text{window}} = \big[u_{t-3}, \; u_{t-2}, \; u_{t-1}, \; u_t\big]$$
   $$u_{\text{conv}, t}[d] = b_{\text{conv}}[d] + \sum_{k=0}^{K-1} W_{\text{conv}}[d, k] \cdot u_{\text{window}}[d, k]$$
   $$u'_t = \text{SiLU}(u_{\text{conv}, t})$$

4. **SSM Intermediate Projections**:
   $$u'_t \in \mathbb{R}^{D_{\text{inner}}} \xrightarrow{W_x} \begin{bmatrix} r_t \\ B_t \\ C_t \end{bmatrix}, \qquad r_t \in \mathbb{R}^R, \quad B_t, C_t \in \mathbb{R}^N$$
   $$\tilde{p}_t = W_{\Delta} r_t + b_{\Delta} \in \mathbb{R}^{D_{\text{inner}}}$$

5. **C1 Recurrence Update**:
   - Retrieve $q_t[d] = \text{sigmoid}(-\tilde{p}_t[d])$ via `mm_lut_q_q31`.
   - Retrieve $\Delta_t[d] = \text{softplus}(\tilde{p}_t[d])$ via `mm_lut_softplus_q24`.
   - Compute modulation vector: $v_t[d] = \Delta_t[d] \cdot u'_t[d] \in \mathbb{R}^{D_{\text{inner}}}$.
   - Update latent state $h$ using successive powers $P_n = q_t[d]^{n+1}$:
     $$h_t[d, n] = P_n \cdot h_{t-1}[d, n] + v_t[d] \cdot B_t[n]$$

6. **Output Projection & Gating**:
   $$y_t[d] = \left(\sum_{n=0}^{N-1} h_t[d, n] \cdot C_t[n]\right) + D_{\text{skip}}[d] \cdot u'_t[d]$$
   $$y_{\text{gate}, t} = y_t \odot \text{SiLU}(z_t)$$
   $$\text{out}_t = \text{token}_t + W_{\text{out}} y_{\text{gate}, t}$$

---

## 4. W8A16 Fixed-Point Quantization & Numerical Stability

To execute natively on integer architectures without floating-point emulation, the model is implemented in fixed-point arithmetic. "W8A16" alone is imprecise, so the exact widths are:
- weights: int8, per-row symmetric scale, requantised with a Q15 multiplier ($M \in [16384, 32767]$) and a shift;
- 16-bit activations: residual Q$r$ (Q6 for the clip-trained baseline, Q5 for the paper model, see below), RMSNorm output Q12, in_proj Q11, conv/SiLU output Q12, x_proj ($\Delta$-rank, $B$, $C$) Q10, $\tilde p$ Q11;
- **32-bit**: SSM state $h$ (Q22), SSM output $y$ (Q12), gated output $y_{\text{gate}}$ (Q12), logits (Q12) and all accumulators; the out_proj accumulator is 64-bit;
- decay powers $P_n$ in Q31, $\Delta$ in Q24.

The residual format is chosen per checkpoint by calibration on validation clips **and** long validation streams (99.99th percentile, 1.25x margin), then by evaluating $q-1, q, q+1$ on the **validation** split (fixed accuracy, then agreement, then lower residual saturation; the format only, never the model, and never test accuracy). Clip-trained baseline: Q6 gave 92.31 % fixed accuracy with 0.002 % residual saturation, Q7 92.27 % with 0.048 % <!-- paper/data/resid_format_choice_shipped.json -->. Paper model (calibrated Q6), validation split: Q5 89.24 %, Q6 and Q7 89.22 % (one clip fewer), residual saturation 0.000 %, 0.000 % and 0.004 % <!-- paper/data/resid_format_choice.json -->; Q5 was exported. (An earlier version selected on test accuracy and chose Q7; see `*_warm800.json`.)

### 4.1 Requantization Arithmetic & Instruction Latency
Linear layer dot products are accumulated in 32-bit signed integers:

$$\text{Acc} = \sum_{i=1}^{M_{\text{in}}} W_{\text{int8}, i} \cdot X_{\text{int16}, i}$$

Since $W \in [-127, 127]$ and $X \in [-32768, 32767]$, each multiply contributes up to $4.16 \times 10^6$. For maximum inner dimension $M_{\text{in}} = 64$:

$$|\text{Acc}| \le 64 \times 127 \times 32768 = 266{,}338{,}304 < 2^{31} - 1 \quad (\mathbf{no\ accumulator\ overflow})$$

To rescale $\text{Acc}$ back to the output fixed-point scale, calibrated integer scaling uses multiplier $M \in [1, 32767]$ and right shift $S \in [0, 31]$. Because $\text{Acc}$ is 32-bit and $M$ is up to 15-bit, their product can reach:

$$|\text{Acc} \times M| \le 266{,}338{,}304 \times 32767 \approx 8.73 \times 10^{12} \approx 2^{43}$$

This strictly exceeds the 32-bit range and **requires an explicit 64-bit intermediate**:

$$\text{Scaled} = \left\lfloor \frac{(\text{int64\_t})\text{Acc} \times M + \text{Round}(S)}{2^S} \right\rfloor$$

where the rounding term is defined as **round-half-up** (ties round toward $+\infty$):

$$\text{Round}(S) = \begin{cases} 2^{S-1} & \text{if } S > 0 \\ 0 & \text{if } S = 0 \end{cases}$$

On RV32IM this 64-bit product is built from `mul`/`mulh` plus a multi-word shift. We did **not** measure the cost of the requantisation sequence in isolation, so no per-requant cycle count is claimed; the measured end-to-end figures are in §6. The rounding is exact: over random $(\text{Acc}, M, S)$, $\max|\text{requant} - \text{Acc}\cdot M/2^S| = 1/2$ (`tests/test_math_claims.py`, T15).

### 4.2 Look-Up Table (LUT) Specification & Domain Clamping
Non-linear activation functions are evaluated via precomputed tables with piecewise linear interpolation:

| Table Identifier | Target Function | Domain | Format | Table Entries | Flash Footprint |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `mm_lut_silu_q12` | $\text{SiLU}(x) = x \cdot \sigma(x)$ | $[-8, 8]$ in Q12 | Q12 signed | $513 \times \text{int16\_t}$ | $1{,}026\text{ bytes}\ (1.00\text{ KiB})$ |
| `mm_lut_q_q31` | $q = \text{sigmoid}(-\tilde{p})$ | $[-16, 16]$ in Q11 | Q31 signed | $513 \times \text{int32\_t}$ | $2{,}052\text{ bytes}\ (2.00\text{ KiB})$ |
| `mm_lut_softplus_q24` | $\Delta = \text{softplus}(\tilde{p})$ | $[-16, 16]$ in Q11 | Q24 signed | $513 \times \text{int32\_t}$ | $2{,}052\text{ bytes}\ (2.00\text{ KiB})$ |
| **Total LUT Footprint** | | | | | **$5{,}130\text{ bytes}$ ($5.01\text{ KiB} \approx 5.13\text{ kB}$)** |

**Domain Representation & Clamping**:
- The raw projection $\tilde{p}$ is represented in **Q11 format** ($LSB = 2^{-11} \approx 4.88 \times 10^{-4}$). The table domain $[-16, 16]$ maps to the integer range $[-32{,}768, 32{,}768]$.
- In `mm_block_q.c`, $\tilde{p}$ is clamped to $\pm (12 \ll 11) = \pm 24{,}576$, fitting cleanly inside 16-bit and 32-bit signed variables.
- There is no explicit clamp on $q$. Contractivity ($q < 1$) holds by construction: every table entry is $\le \sigma(16)(2^{31}-1) < 2^{31}-1$ and linear interpolation cannot exceed its end points. Because $\tilde p$ is clamped to $\pm 12$ while the tables span $\pm 16$, the outer $2 \times 64$ entries of each $\tilde p$ table are never read (they are kept so the table domain is a power of two).

Linear interpolation evaluates via:
$$\text{diff} = x - x_{\text{min}}, \qquad \text{idx} = \text{diff} \gg \text{step\_shift}, \qquad \text{frac} = \text{diff} \ \& \ \big((1 \ll \text{step\_shift}) - 1\big)$$
$$f(x) = \text{LUT}[\text{idx}] + \left\lfloor \frac{(\text{LUT}[\text{idx}+1] - \text{LUT}[\text{idx}]) \times \text{frac} + 2^{\text{step\_shift}-1}}{2^{\text{step\_shift}}} \right\rfloor$$

The cycle cost of this routine was not measured separately and no instruction count is claimed.

### 4.3 Error Analysis: Tight Bounds & Relative Decay Yardstick
Because decay powers $P_n = q^{n+1}$ are computed by repeated multiplication, an initial table lookup error $\delta q$ propagates according to the derivative:

$$\frac{d}{dq} (q^{n+1}) = (n + 1) q^n \implies \delta(q^{n+1}) \approx (n + 1) q^n \cdot \delta q$$

1. **Interpolation Error Peak**:
   Linear interpolation truncation error is bounded by $|\delta q| \le \frac{h^2}{8} \max |f''(x)|$. For $f(x) = \text{sigmoid}(-x)$, the second derivative evaluates to $f''(x) = \sigma(-x)(1 - \sigma(-x))(1 - 2\sigma(-x))$.
   This derivative does **not** peak near $q \to 1$; rather, it peaks at $q \approx 0.211$ and $q \approx 0.789$, where $|f''(x)|_{\max} = \frac{1}{6\sqrt{3}} \approx 0.0962$.
   At $q \approx 0.789$, the amplification factor $(n + 1) q^n$ is largest for $n = 3$, where it equals **$\approx 1.96$** (computed numerically over $n = 0..7$; well below the naive upper bound of 8).
   Conversely, as $q \to 1^-$ (where the naive multiplier $n+1 \to 8$ peaks), $\sigma''(x) \to 0$ exponentially fast, causing the base interpolation error $\delta q$ to vanish.

2. **Relative Error in the Decay Rate**:
   In recurrent state propagation, $P_n$ multiplies the latent state $h_{t-1}$. The appropriate physical yardstick is not absolute additive error against an activation LSB, but the **relative error in the continuous-time decay rate** $\alpha_n = 1 - P_n \approx (n + 1)\Delta$:
   $$\frac{|\delta(1 - P_n)|}{1 - P_n} = \frac{|\delta P_n|}{1 - P_n} \approx \frac{(n + 1) q^n |\delta q|}{(n + 1)(1 - q)} = \frac{q^n |\delta q|}{1 - q}$$
   As $q \to 1^-$, $1 - q = \text{sigmoid}(\tilde{p}) \approx e^{\tilde{p}}$, while truncation error scales as $\delta q \le \frac{h^2}{8} |f''(\tilde{p})| \approx \frac{h^2}{8} e^{\tilde{p}}$. The exponential terms cancel, yielding:
   $$\frac{|\delta(1 - P_n)|}{1 - P_n} \lesssim \frac{h^2}{8} = \frac{(0.0625)^2}{8} \approx 4.88 \times 10^{-4}$$
   is the leading-order estimate, but it is **not a strict bound**. Over a cell the error peak can sit where $f''$ is larger than at the cell's end by up to a factor $e^{h}$ ($\approx 1.065$), and the Q31 rounding of the table entries adds $2^{-31}/(1-q)$ ($\approx 7.6 \times 10^{-5}$ at the clamp edge $1-q = 6.14 \times 10^{-6}$). Measured over all $\tilde p$ in Q11 steps: **$\max |\delta q|/(1-q) = 5.97 \times 10^{-4}$ at $\tilde p = -11.97$** (about 22 % above the leading-order value), and the repeated-multiplication powers $P_n$ stay within $6.13 \times 10^{-4}$ for all $n$ <!-- paper/data/math_claims_final.log T6, T7; paper/data/math_numbers.json -->. The relative error of the decay rate is therefore $\lesssim 0.06\%$.

3. **Input Quantization Contribution**:
   Input $\tilde{p}$ is quantized to Q11 ($LSB = 2^{-11}$). The maximum rounding error of $\frac{1}{2} LSB$ contributes an input-induced error of $|\delta q|_{\text{input}} \le \sigma'(x) \cdot 2^{-12} \le 0.25 \times 2.44 \times 10^{-4} \approx 6.1 \times 10^{-5}$, which remains strictly bounded and contractive.

### 4.4 Latent State Precision, Floor Truncation and the Shipped Rounding
The state update is

$$h_t[d, n] = \left\lfloor \frac{(\text{int64\_t})P \cdot h_{t-1}[d, n] + 2^{30}}{2^{31}} \right\rfloor + v_t[d] \cdot B_t[n]$$

with $h$ stored as int32 in Q22. **The shipped code adds $2^{30}$, i.e. it rounds half up.** The motivation is the following property of the plain arithmetic shift (floor):

1. **Floor truncation is biased.** For $h > 0$, $\lfloor P h / 2^{31} \rfloor \le h - 1$, so the state always loses at least one LSB per step. For $h = -k < 0$ and $P = 2^{31} - \Delta_P$, $\lfloor -Pk/2^{31} \rfloor = -k + \lfloor \Delta_P k / 2^{31} \rfloor$, which equals $-k$ whenever $|h| < 1/(1-P)$ LSB: negative states do not decay at all. In a noise experiment at $1-P = 10^{-3}$ the floor update drifts to a mean of $-247$ LSB in the last 10,000 steps, against $-55$ LSB for the rounded update, averaged over 16 seeds (the rounded value lies within the random-walk spread of the experiment) <!-- paper/data/math_numbers.json rounding_drift; tests/test_math_claims.py T11, T12 -->.
2. **Rounding leaves a symmetric dead band.** With round-half-up the update stops changing the state when $|h| < 0.5/(1-P)$ LSB, on both sides of zero. At $1-P = 10^{-3}$ this is 500 LSB $= 1.19 \times 10^{-4}$ in real units (the floor update's one-sided band is 1,000 LSB $= 2.38 \times 10^{-4}$). At the clamp edge $1-q = 6.14 \times 10^{-6}$ (the slowest channels, a time constant of about 160,000 tokens) the band is 81,400 LSB $= 0.0194$ in real units, comparable to the typical magnitude of a quiet state; those channels behave as near-integrators with a small dead zone. About 1 % of the clip-trained baseline's channel-steps on long streams sit at that edge; for the paper model the 1st percentile of $1-q$ on streams is $7.1 \times 10^{-5}$ and the clamp is not reached <!-- paper/data/activation_ranges_shipped.json, activation_ranges_final.json -->.
3. **Headroom.** The state magnitude reached in float evaluation is at most 54 (clips) and 35 (streams) for the paper model, and 41 and 33 for the clip-trained baseline, against a Q22 limit of 512, so about 3.5 bits are unused and the int32 state never saturated in the evaluation (`h` counter: 0 of $2.5 \times 10^{8}$ updates for both) <!-- paper/data/activation_ranges_final.json, activation_ranges_shipped.json, eval_full_test_final.json -->.

---

## 5. Working Memory Footprint: $\mathcal{O}(1)$ Streaming vs. $\mathcal{O}(T)$ Batch

### 5.1 Exact Static Context of KWS-Mu-Mamba
The persistent state is statically allocated in `mm_context_t` (the host float reference state is compiled out of the firmware build). Field by field:

| Field | Bytes |
| :--- | ---: |
| `audio_buf` int16[512] | 1,024 |
| `audio_buf_count` uint16 | 2 |
| `mel_features` int16[40] | 80 |
| conv ring int16[2][3][64] | 768 |
| alignment padding | 2 |
| SSM state int32[2][64][8] | 4,096 |
| `logits_q` int32[12] | 48 |
| `smoothed_margins` int32[12] | 48 |
| `refractory_timer` int16 + padding | 4 |
| **`sizeof(mm_context_t)`** | **6,072** |

The recurrent state alone ($h$ and the conv ring) is $4{,}096 + 768 = 4{,}864$ B. Inference also uses stack: a task that only runs `mm_step_q` used **3,680 B** of stack in the measurement of 2026-10-30 (3,680–3,696 B on 2026-10-11; FFT real/imaginary arrays $2 \times 1{,}024$ B, `power[257]` uint32 $1{,}028$ B, block scratch). Heap allocated during inference was **0 B** (free and minimum free heap identical, 300,500 B, before and after 51 steps). Static context plus stack is therefore $6{,}072 + 3{,}680 = 9{,}752$ B <!-- paper/data/board_bench_final.json -->. Earlier versions of this document gave 6,066 B and 98 B for the logit/decision state, which omitted the 2-byte sample count, the padding and `power[257]`.

### 5.2 Offline-Window Baseline: What Can Be Said From Its Source
An offline-window implementation with $T = 100$, $D_{\text{inner}} = 128$, $N = 16$ keeps whole-sequence activations alive. Reading its SSM loop, the concurrently live float matrices are `x`, `z`, `dt`, `y_out` ($128 \times 100 \times 4$ B $= 51{,}200$ B each), `x_ssm` and `A` ($8{,}192$ B each), `B` and `C` ($6{,}400$ B each):

$$51{,}200 \times 4 + 8{,}192 \times 2 + 6{,}400 \times 2 = 233{,}984\ \text{B} \;(228.5\ \text{KiB}).$$

This is an arithmetic consequence of the tensor shapes (`tests/test_math_claims.py`, T14), **not a measurement**: the heap peak, leak and `malloc` behaviour of that code on the ESP32-C3 were not measured in this work, and earlier hardware figures quoted for it in this repository were unverified and are withdrawn. For comparison, the recurrent state here is $4{,}864$ B and the whole static context $6{,}072$ B. The memory advantage is a property of the $\mathcal{O}(1)$ streaming topology, not of the C1 recurrence.

---

## 6. Where the Speed Comes From (Measured, and What Is Still Open)

### 6.1 Same model, soft-float against fixed point (measured on the ESP32-C3 @ 160 MHz, paper model)
Both rows run the same trained network (the paper model, export `6a9a3c92`) and the same integer front-end, measured with the on-board cycle counter (median of repeated steps on a fixed noise frame), build of 2026-10-30 <!-- paper/data/board_bench_final.json -->:

| Per-token step | Cycles | Time |
| :--- | ---: | ---: |
| Front-end only (Hann, FFT, mel, log) | 130,325 | 0.815 ms |
| `mm_step_q`: front-end + fixed-point model | 662,595 | 4.141 ms |
| `mm_step_f32`: front-end + FP32 soft-float model | 7,238,407 | 45.24 ms |

The fixed-point engine is **10.9x** faster than the FP32 soft-float implementation of the same model ($7{,}238{,}407 / 662{,}595$), and runs at a real-time factor of $4.141/20 = 0.207$ against a 20 ms hop (the float build is at 2.26 and cannot keep up). Micro-costs measured on the same core: `expf` 2,831 cycles, soft-float MAC 227 cycles, int8 x int16 MAC 9.0 cycles (loop and loads included). The fixed model costs about $662{,}595 - 130{,}325 = 532{,}270$ cycles for about $22 \times 10^3$ multiplies, i.e. about 24 cycles per multiply, which includes LUT interpolation, requantisation and cache effects; the flash cache is 16 KB while roughly 26 KB of weights and tables are read per token, so cache misses probably contribute, but this was not measured. Over the serial parity run (50,000 tokens) the device reported a median of 665,300 cycles per token including the serial protocol's own overhead, and the 50,000 logits vectors were bit-identical to the host engine <!-- paper/data/serial_eval_final.json -->. An earlier measurement of the clip-trained model on an earlier export gave 652,719 cycles (`paper/data/board_bench_pre_fix.json`); the 1.5 % difference was not isolated.

### 6.2 The C1 recurrence in isolation (measured)
The FP32 soft-float full step (front-end + model) was timed with the C1 power recurrence $P_n = P_{n-1} q$ and with a per-$(d,n)$ decay $\exp(-\Delta(n+1))$ computed with `expf`, on the same input, 11 rounds after a warm-up <!-- paper/data/board_c1_ablation_final.json -->. The two variants are the same model: the maximum absolute logit difference over the rounds was $2.7 \times 10^{-6}$.

| Variant | Cycles per step (median) | Time |
| :--- | ---: | ---: |
| C1 power recurrence | 7,239,491 | 45.25 ms |
| per-$(d,n)$ `expf` decay | 9,147,942 | 57.17 ms |

The `expf` form is **1.26x** slower, an extra **1,908,451 cycles per token**. The estimate given before the measurement (7 extra `expf` $\times$ 2,831 cycles $\times$ 128 channel-layers $\approx 2.5 \times 10^6$) overestimated it by about 30 %; the measurement supersedes it. This is the saving of the recurrence in the soft-float build. The fixed-point engine calls no `expf`, so the corresponding saving there was not separately measured.

### 6.3 What is not claimed
No percentage attribution of the speed-up to individual optimisations (soft-float removal, C1, LUTs, streaming) is claimed: earlier percentages were estimates and are withdrawn; the measured quantities are the 10.9x fixed-vs-soft-float ratio and the 1.26x C1-vs-`expf` ratio above. The third-party offline-window baseline (MambaLite-Micro) was built from its published repository (all library sources byte-identical to commit `44d51fce`) and run on the same board, but it did not give consistent repeated measurements: the first call took 9.74 s, a second call on the same input 0.46 s with a different prediction, and the third call crashed after the free heap had fallen to 94,160 B (minimum seen 10,376 B) <!-- paper/data/baseline_mambalite_c3.json -->. No speed or memory ratio against it is therefore claimed. Its model also differs from ours (35,914 parameters, 10 classes, a whole 100-frame window against 18,060 parameters, 12 classes and a 20 ms hop). Statements about the C3 memory system beyond the facts above (cache hit rates, wait states) are likewise unmeasured.

### 6.4 Reading streaming detection numbers
The decision rule smooths the margin of each keyword logit over the larger of the two background logits, $m_k = z_k - \max(z_{\text{unknown}}, z_{\text{silence}})$, with an EMA $s_t = \alpha s_{t-1} + (1-\alpha) m_t$ (integer, Q15 $\alpha$), fires when the best $s_t$ exceeds a threshold, and then suppresses the next $R$ tokens (refractory period). It is a margin, not a probability. Engine defaults: $\alpha = 0.93$, $R = 75$ tokens (1.5 s), threshold 1.5 (6144 in Q12): the setting the validation-only tuner chose for the paper model's fixed engine under the false-alarm budget; it is also the best of the extended grid $\alpha \in \{0.85, 0.93, 0.96, 0.98\} \times R \in \{50, 75, 100, 150\}$, so it is no longer at the edge of the search <!-- paper/data/stream_final.json -->.

Two accounting points: (i) **false alarms per background hour** divide by the stream time that contains no keyword clip and **include substitutions** (a wrong keyword fired inside a keyword's window); pure-background false alarms and substitutions are reported separately by `stream_eval`; (ii) a **detection rate is only meaningful together with its false-alarm rate**. At the lowest threshold of the tuning grid (0.25) the clip-trained baseline fires about 1.7 times per background second (9,484 false alarms in 1.547 background hours, `paper/data/stream_shipped.json`, produced with the earlier rule $\alpha = 0.7$, $R = 25$ and the earlier tuner), so most "detections" are class guesses that happen to fall inside a keyword window; the 91.3 % detection on the test stream and the 70.5 % on the validation stream at that threshold differ only because the guesses land differently in the two streams. False-alarm rates are given with an exact (Garwood) Poisson 95 % interval of the count divided by the observation time, e.g. 14 pure-background false alarms in 1.547 background hours are $9.0$ per hour with interval $[4.9, 15.2]$ (counts are small, so the intervals are wide). Operating points are therefore selected under a false-alarm budget (`stream_eval --fa-budget`, default 100 per background hour) rather than by a detection target.
