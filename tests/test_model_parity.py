import torch
import numpy as np
from train.kws.model import MambaC1Block, MuMambaKWS

def test_check_1_forward_vs_step():
    """Check 1: PyTorch forward(seq) vs step() loop: max abs diff < 1e-5"""
    torch.manual_seed(42)
    B, T, D = 2, 50, 32
    block = MambaC1Block(d_model=D, d_inner=64, d_state=8, dt_rank=4, d_conv=4)
    block.eval()

    x_seq = torch.randn(B, T, D)

    # 1. Batched forward(seq)
    with torch.no_grad():
        out_forward = block(x_seq) # (B, T, D)

    # 2. Sequential step() loop
    conv_ring = torch.zeros(B, block.d_conv - 1, block.d_inner)
    h = torch.zeros(B, block.d_inner, block.d_state)
    state = (conv_ring, h)

    out_step_list = []
    with torch.no_grad():
        for t in range(T):
            token = x_seq[:, t, :] # (B, D)
            out_token, state = block.step(token, state)
            out_step_list.append(out_token)
    out_step = torch.stack(out_step_list, dim=1) # (B, T, D)

    max_diff = torch.max(torch.abs(out_forward - out_step)).item()
    print(f"Check 1: MambaC1Block forward(seq) vs step() max abs diff = {max_diff:.3e}")
    assert max_diff < 1e-5, f"Check 1 failed: max diff {max_diff} >= 1e-5"

    # Test full model MuMambaKWS forward vs step
    model = MuMambaKWS()
    model.eval()
    mel_input = torch.randn(B, T, 40)
    with torch.no_grad():
        logits_forward = model(mel_input)

    states = model.init_streaming_state(batch_size=B)
    with torch.no_grad():
        for t in range(T):
            logits_step, states = model.step(mel_input[:, t, :], states)

    model_diff = torch.max(torch.abs(logits_forward - logits_step)).item()
    print(f"Check 1 (Full Model): forward vs step final logits max abs diff = {model_diff:.3e}")
    assert model_diff < 1e-5, f"Check 1 (Full Model) failed: max diff {model_diff} >= 1e-5"
    print("Check 1 PASSED!")

def test_check_2_exp_scan_vs_c1_recurrence():
    """
    Check 2: Generic exp(Delta * A) scan vs C1 power recurrence, same weights
    Pass criterion: max abs diff < 1e-5
    """
    torch.manual_seed(42)
    B, T, D_i, N = 2, 50, 64, 8

    # A matrix fixed to -(n + 1) for n = 0..N-1
    # Shape: (1, 1, 1, N)
    A_exact = -torch.arange(1, N + 1, dtype=torch.float32).view(1, 1, 1, N)

    p = torch.randn(B, T, D_i)
    p_clamped = torch.clamp(p, -12.0, 12.0)
    Delta = torch.nn.functional.softplus(p_clamped) # (B, T, D_i)
    u_prime = torch.randn(B, T, D_i)
    v = Delta * u_prime
    B_val = torch.randn(B, T, N)

    # 1. Generic Mamba scan using exp(Delta * A):
    # Delta * A: (B, T, D_i, 1) * (1, 1, 1, N) -> (B, T, D_i, N)
    exp_dA = torch.exp(Delta.unsqueeze(-1) * A_exact) # (B, T, D_i, N)
    dB = v.unsqueeze(-1) * B_val.unsqueeze(-2)         # (B, T, D_i, N)

    h_exp = torch.zeros(B, D_i, N)
    for t in range(T):
        h_exp = exp_dA[:, t] * h_exp + dB[:, t]

    # 2. C1 power recurrence:
    # q = sigmoid(-p) = exp(-Delta)
    q = torch.sigmoid(-p_clamped) # (B, T, D_i)
    P_powers = [q]
    for _ in range(1, N):
        P_powers.append(P_powers[-1] * q)
    P_c1 = torch.stack(P_powers, dim=-1) # (B, T, D_i, N)

    # Verify that exp(Delta * A) and P_c1 are identical
    exp_vs_power_diff = torch.max(torch.abs(exp_dA - P_c1)).item()
    print(f"Check 2 (P matrix): exp(Delta * A) vs C1 P matrix max abs diff = {exp_vs_power_diff:.3e}")
    assert exp_vs_power_diff < 1e-5, f"Check 2 P matrix diff {exp_vs_power_diff} >= 1e-5"

    h_c1 = torch.zeros(B, D_i, N)
    for t in range(T):
        h_c1 = P_c1[:, t] * h_c1 + dB[:, t]

    state_diff = torch.max(torch.abs(h_exp - h_c1)).item()
    print(f"Check 2 (State h): exp(Delta * A) vs C1 scan max abs diff = {state_diff:.3e}")
    assert state_diff < 1e-5, f"Check 2 failed: state diff {state_diff} >= 1e-5"

    # 3. Model code verification: compare generic exp(dA) scan vs MambaC1Block
    block = MambaC1Block(d_model=32, d_inner=D_i, d_state=N, dt_rank=4, d_conv=4)
    block.eval()
    x_in = torch.randn(B, 32)
    # Step block using C1 recurrence
    conv_ring = torch.zeros(B, block.d_conv - 1, block.d_inner)
    h_c1_layer = torch.zeros(B, block.d_inner, block.d_state)
    out_c1, (conv_ring, h_c1_layer) = block.step(x_in, (conv_ring, h_c1_layer))

    # Generic exp(dA) scan on the same layer:
    x_norm = block.norm(x_in)
    xz = block.in_proj(x_norm)
    u, z = xz.chunk(2, dim=-1)
    init_ring = torch.zeros(B, block.d_conv - 1, block.d_inner)
    conv_input = torch.cat([init_ring, u.unsqueeze(1)], dim=1)
    u_prime = torch.nn.functional.silu(block.conv1d(conv_input.transpose(1, 2)).squeeze(-1))
    x_ssm = block.x_proj(u_prime)
    dt_rank, B_proj, C_proj = torch.split(x_ssm, [block.dt_rank, block.d_state, block.d_state], dim=-1)
    p_layer = torch.clamp(block.dt_proj(dt_rank), -12.0, 12.0)
    delta_layer = torch.nn.functional.softplus(p_layer)
    v_layer = delta_layer * u_prime

    # Generic scan step:
    exp_dA_layer = torch.exp(delta_layer.unsqueeze(-1) * A_exact.squeeze(0))
    dB_layer = v_layer.unsqueeze(-1) * B_proj.unsqueeze(-2)
    h_exp_layer = exp_dA_layer * torch.zeros(B, D_i, N) + dB_layer
    y_exp = torch.sum(h_exp_layer * C_proj.unsqueeze(-2), dim=-1) + (block.D_skip * u_prime)
    y_gate_exp = y_exp * torch.nn.functional.silu(z)
    out_exp = x_in + block.out_proj(y_gate_exp)

    layer_diff = torch.max(torch.abs(out_c1 - out_exp)).item()
    print(f"Check 2 (Model Layer): Generic exp(Delta * A) vs MambaC1Block max diff = {layer_diff:.3e}")
    assert layer_diff < 1e-5, f"Check 2 Model Layer failed: max diff {layer_diff} >= 1e-5"
    print("Check 2 PASSED!")

if __name__ == "__main__":
    print("=== Testing Verification Checks 1 & 2 ===")
    test_check_1_forward_vs_step()
    test_check_2_exp_scan_vs_c1_recurrence()
    print("All Model Parity Checks Passed Successfully!")
