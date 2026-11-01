# Exponential-Free Mamba: Integer-Only Streaming Keyword Spotting on an FPU-less Microcontroller

## Abstract

Mamba-style state space models (SSMs) stream audio in constant memory, which suits always-on keyword spotting (KWS),
but their recurrence needs an exponential per channel and state, costly on microcontrollers (MCUs) without a
floating-point unit. We present μ-Mamba, an integer-only streaming keyword spotter. Freezing the state matrix to
A = −(n+1) makes the zero-order-hold decay exactly σ(−p)^(n+1), so all decays of a channel follow from one table lookup
and N−1 fixed-point multiplications. We quantise an 18,060-parameter model to int8 weights, 16-bit activations and
32-bit states, and show that clip-trained models fail on continuous audio because their state never returns to
background; fine-tuning on long streams fixes this. The integer engine reaches 89.0 % accuracy on 12-class Speech
Commands v2 and detects 79.9 % of keywords in continuous streams at 47 false alarms per background hour. On a 160 MHz
ESP32-C3 it needs 4.14 ms per 20 ms token and 9.75 KB of RAM, is bit-exact to its host reference, and runs 10.9× faster
than software floating point.

<!-- Word count 166 (wc -w). Every number: paper/RESULTS.md §2–§5.
MambaLite-Micro same-chip result deliberately omitted (user decision, 2026-10-30).
Cut from the 275-word version: 3.3-pt cost of stream fine-tuning, 9 pure-background FAs, 10.6 % clip-baseline figure,
fixed-vs-float 0.1 pt gap, error-analysis numbers, frozen-vs-learned-A ablation (3.8 ± 2.1 pt, p = 0.086). -->
