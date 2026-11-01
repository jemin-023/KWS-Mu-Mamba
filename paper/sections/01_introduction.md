## 1 Introduction

Always-on keyword spotting (KWS) runs continuously on small, cheap devices, many of which, such as the RISC-V ESP32-C3, have no floating-point unit (FPU). Convolutional networks dominate this setting [Hello Edge]. Selective state space models (SSMs) such as Mamba [Gu & Dao] are an appealing alternative: they consume audio one frame at a time and carry a fixed-size state, so memory does not grow with the stream. Mamba has been applied to KWS [Keyword Mamba] and, quantised, to microcontrollers (MCUs) [FEMBA], but its discretisation needs exp(ΔA) for every channel and state at every step, and in software floating point one `expf` costs 2,831 cycles on our target. Our model would need 1,024 of them per 20 ms token, against a budget of 3.2 million cycles at 160 MHz.

We ask how far a streaming Mamba keyword spotter can run on an FPU-less MCU with integer arithmetic alone. Our contributions are:

1. **The C1 recurrence.** Freezing the state matrix to A[d,n] = −(n+1) makes the zero-order-hold decay exactly σ(−p)^(n+1). All state decays of a channel then follow from one table lookup and N−1 fixed-point multiplications, with no transcendental function at run time. We analyse its fixed-point error.
2. **An integer-only engine.** An 18,060-parameter model with int8 weights, 16-bit activations and 32-bit states reaches 89.0 % accuracy on 12-class Speech Commands v2 and is bit-exact to its host reference on the device.
3. **A streaming failure and its fix.** A model trained on isolated clips never returns to background after a keyword: background accuracy is 1–20 % for up to 3 s afterwards. Fine-tuning on long multi-clip streams, with a longer refractory period, gives 79.9 % keyword detection at 47 false alarms per background hour, against 10.6 % for the clip-trained model.
4. **On-device measurements.** On an ESP32-C3 at 160 MHz the engine needs 4.14 ms per 20 ms token and 9.75 KB of RAM, and is 10.9× faster than the same model in software floating point.

We also report the limits. Most of the speed-up comes from integer arithmetic, not from C1, which is only 1.26× faster than per-state exponentials in a float build. Freezing A cost 3.8 ± 2.1 accuracy points against a learned A in a three-seed ablation (p = 0.086). Accuracy is below that of CNN keyword spotters, and 47 false alarms per hour is not a production operating point.
