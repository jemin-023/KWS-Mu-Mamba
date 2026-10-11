import math
from typing import Tuple, Optional, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # RMSNorm: x / sqrt(mean(x^2) + eps) * weight
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        return x * torch.rsqrt(variance + self.eps) * self.weight

class MambaC1Block(nn.Module):
    """
    Streaming Mamba Block with C1 recurrence constraint:
      - Fixed state matrix A[d, n] = -(n + 1), left untrained
      - All N decay coefficients come from one scalar q = sigmoid(-p)
        and Delta = softplus(p), by repeated multiplication P_n = P_{n-1} * q.
      - Conv ring buffer carries state for causal depthwise Conv1d (K=4)
      - Batched forward(seq) and streaming step(token, state)
    """
    def __init__(
        self,
        d_model: int = 32,      # D
        d_inner: int = 64,      # D_i
        d_state: int = 8,       # N
        dt_rank: int = 4,       # R
        d_conv: int = 4,        # K
        p_clamp_range: float = 12.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_inner = d_inner
        self.d_state = d_state
        self.dt_rank = dt_rank
        self.d_conv = d_conv
        self.p_clamp_range = p_clamp_range

        self.norm = RMSNorm(d_model)

        # in_proj: projects D -> 2 * D_i (x and z)
        self.in_proj = nn.Linear(d_model, 2 * d_inner, bias=False)

        # Depthwise Conv1d: kernel size K=4, groups=D_inner
        self.conv1d = nn.Conv1d(
            in_channels=d_inner,
            out_channels=d_inner,
            kernel_size=d_conv,
            groups=d_inner,
            bias=True,
            padding=0,
        )

        # x_proj: projects u' (D_inner) -> (dt_rank + 2 * d_state) = 4 + 8 + 8 = 20
        self.x_proj = nn.Linear(d_inner, dt_rank + 2 * d_state, bias=False)

        # dt_proj: projects dt_rank -> D_inner with bias
        self.dt_proj = nn.Linear(dt_rank, d_inner, bias=True)

        # out_proj: projects D_inner -> D
        self.out_proj = nn.Linear(d_inner, d_model, bias=False)

        # D_skip: residual skip connection per inner channel (D_i)
        self.D_skip = nn.Parameter(torch.ones(d_inner))

        # Initialize dt_proj bias to put Delta in typical initial range [1e-3, 1e-1]
        self._init_dt()

    def _init_dt(self):
        # Initialize dt_proj bias so softplus(p) is distributed in [0.001, 0.1]
        dt_min, dt_max = 0.001, 0.1
        dt_init = torch.exp(
            torch.rand(self.d_inner) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        )
        # Inverse softplus: p = log(exp(dt) - 1)
        p_init = torch.log(torch.exp(dt_init) - 1.0)
        with torch.no_grad():
            self.dt_proj.bias.copy_(p_init)
            # Small weights for dt_proj
            self.dt_proj.weight.zero_()

    def _c1_scan(
        self,
        u_prime: torch.Tensor,   # (B, T, D_i)
        p: torch.Tensor,         # (B, T, D_i)
        B_val: torch.Tensor,     # (B, T, N)
        C_val: torch.Tensor,     # (B, T, N)
        h_init: Optional[torch.Tensor] = None, # (B, D_i, N)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Batched C1 scan over time T:
          p is clamped to [-p_clamp_range, p_clamp_range]
          q = sigmoid(-p)
          Delta = softplus(p)
          v = Delta * u_prime
          P_n = P_{n-1} * q  (with P_0 = q)
          h_t = P * h_{t-1} + v_t (x) B_t
          y_t = h_t @ C_t
        """
        B_sz, T, _ = u_prime.shape
        p_clamped = torch.clamp(p, -self.p_clamp_range, self.p_clamp_range)

        # Stable softplus and sigmoid
        Delta = F.softplus(p_clamped)          # (B, T, D_i)
        q = torch.sigmoid(-p_clamped)          # (B, T, D_i)
        v = Delta * u_prime                    # (B, T, D_i)

        # Precompute decay powers P: (B, T, D_i, N)
        # P[:, :, :, 0] = q, P[:, :, :, n] = P[:, :, :, n-1] * q
        P_powers = [q]
        for _ in range(1, self.d_state):
            P_powers.append(P_powers[-1] * q)
        P = torch.stack(P_powers, dim=-1)      # (B, T, D_i, N)

        # Outer product: v (B, T, D_i, 1) * B_val (B, T, 1, N) -> (B, T, D_i, N)
        dB = v.unsqueeze(-1) * B_val.unsqueeze(-2)

        # Recurrent scan over T steps
        if h_init is None:
            h = torch.zeros(B_sz, self.d_inner, self.d_state, device=u_prime.device, dtype=u_prime.dtype)
        else:
            h = h_init.clone()

        y_list = []
        for t in range(T):
            P_t = P[:, t]      # (B, D_i, N)
            dB_t = dB[:, t]    # (B, D_i, N)
            C_t = C_val[:, t]  # (B, N)

            # Recurrence: h = P * h + dB
            h = P_t * h + dB_t

            # Output: y = sum_n (h[:, d, n] * C[:, n])
            # (B, D_i, N) @ (B, N, 1) -> (B, D_i)
            y_t = torch.bmm(h, C_t.unsqueeze(-1)).squeeze(-1)
            y_list.append(y_t)

        y = torch.stack(y_list, dim=1) # (B, T, D_i)
        return y, h

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Batched forward:
          x: (B, T, D)
        Returns:
          out: (B, T, D)
        """
        B_sz, T, D = x.shape
        residual = x

        # 1. RMSNorm
        x_norm = self.norm(x)

        # 2. in_proj
        xz = self.in_proj(x_norm) # (B, T, 2 * D_inner)
        u, z = torch.split(xz, [self.d_inner, self.d_inner], dim=-1)

        # 3. Causal Conv1d on u:
        # Pad left by (K - 1) tokens
        u_trans = u.transpose(1, 2) # (B, D_inner, T)
        u_padded = F.pad(u_trans, (self.d_conv - 1, 0))
        u_conv = self.conv1d(u_padded) # (B, D_inner, T)
        u_prime = F.silu(u_conv.transpose(1, 2)) # (B, T, D_inner)

        # 4. x_proj
        x_ssm = self.x_proj(u_prime) # (B, T, R + 2N)
        dt_rank_val, B_val, C_val = torch.split(
            x_ssm, [self.dt_rank, self.d_state, self.d_state], dim=-1
        )

        # 5. dt_proj -> p
        p = self.dt_proj(dt_rank_val) # (B, T, D_inner)

        # 6. C1 Scan
        y, _ = self._c1_scan(u_prime, p, B_val, C_val)
        y = y + self.D_skip.view(1, 1, -1) * u_prime

        # 7. Gating with z
        y_gate = y * F.silu(z)

        # 8. out_proj
        out = self.out_proj(y_gate)

        return residual + out

    def step(
        self,
        token: torch.Tensor,
        state: Tuple[torch.Tensor, torch.Tensor],
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Streaming step for a single token:
          token: (B, D)
          state: (conv_ring, h)
            - conv_ring: (B, K - 1, D_inner)
            - h: (B, D_inner, N)
        Returns:
          out_token: (B, D)
          new_state: (new_conv_ring, new_h)
        """
        conv_ring, h = state
        residual = token

        # 1. RMSNorm
        x_norm = self.norm(token)

        # 2. in_proj
        xz = self.in_proj(x_norm)
        u, z = torch.split(xz, [self.d_inner, self.d_inner], dim=-1)

        # 3. Conv step with ring buffer
        # Concatenate ring buffer of (K-1) tokens with current token u (1 token)
        # Ring has shape (B, K-1, D_inner), u has (B, 1, D_inner)
        u_window = torch.cat([conv_ring, u.unsqueeze(1)], dim=1) # (B, K, D_inner)
        new_conv_ring = u_window[:, 1:, :] # Drop oldest, keep (K-1) tokens

        # Apply conv weights:
        # conv1d.weight is (D_inner, 1, K)
        # Dot product across K taps for each channel
        u_conv = F.conv1d(
            u_window.transpose(1, 2), # (B, D_inner, K)
            self.conv1d.weight,
            self.conv1d.bias,
            groups=self.d_inner,
        ).squeeze(-1) # (B, D_inner)

        u_prime = F.silu(u_conv) # (B, D_inner)

        # 4. x_proj
        x_ssm = self.x_proj(u_prime) # (B, R + 2N)
        dt_rank_val, B_val, C_val = torch.split(
            x_ssm, [self.dt_rank, self.d_state, self.d_state], dim=-1
        )

        # 5. dt_proj -> p
        p = self.dt_proj(dt_rank_val) # (B, D_inner)
        p_clamped = torch.clamp(p, -self.p_clamp_range, self.p_clamp_range)

        # 6. C1 Recurrence step
        q = torch.sigmoid(-p_clamped) # (B, D_inner)
        Delta = F.softplus(p_clamped) # (B, D_inner)
        v = Delta * u_prime           # (B, D_inner)

        # Repeated multiplication for P
        P_powers = [q]
        for _ in range(1, self.d_state):
            P_powers.append(P_powers[-1] * q)
        P = torch.stack(P_powers, dim=-1) # (B, D_inner, N)

        # Input outer product
        dB = v.unsqueeze(-1) * B_val.unsqueeze(-2) # (B, D_inner, N)

        # State update
        new_h = P * h + dB # (B, D_inner, N)

        # Output
        y = torch.bmm(new_h, C_val.unsqueeze(-1)).squeeze(-1) # (B, D_inner)
        y = y + self.D_skip.view(1, -1) * u_prime

        # 7. Gating
        y_gate = y * F.silu(z)

        # 8. out_proj
        out = self.out_proj(y_gate)

        return residual + out, (new_conv_ring, new_h)


class MuMambaKWS(nn.Module):
    """
    Complete Keyword Spotter Model:
      - Stem: Linear (40 log-mel -> D)
      - Blocks: L x MambaC1Block
      - Head: RMSNorm(D) -> Linear(D -> 12)
    """
    def __init__(
        self,
        in_dim: int = 40,
        d_model: int = 32,
        d_inner: int = 64,
        d_state: int = 8,
        dt_rank: int = 4,
        d_conv: int = 4,
        num_layers: int = 2,
        num_classes: int = 12,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_classes = num_classes

        # Stem projection: 40 -> D
        self.stem = nn.Linear(in_dim, d_model)

        # Residual Mamba blocks
        self.layers = nn.ModuleList([
            MambaC1Block(
                d_model=d_model,
                d_inner=d_inner,
                d_state=d_state,
                dt_rank=dt_rank,
                d_conv=d_conv,
            )
            for _ in range(num_layers)
        ])

        # Final head
        self.final_norm = RMSNorm(d_model)
        self.head = nn.Linear(d_model, num_classes)

    def forward_seq(self, x: torch.Tensor) -> torch.Tensor:
        """
        Batched sequence forward:
          x: (B, T, 40)
        Returns:
          logits: (B, T, 12) for all tokens T
        """
        h = self.stem(x) # (B, T, D)
        for layer in self.layers:
            h = layer(h)
        logits = self.head(self.final_norm(h)) # (B, T, 12)
        return logits

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Batched forward:
          x: (B, T, 40)
        Returns:
          logits: (B, 12) read at the last token T
        """
        return self.forward_seq(x)[:, -1, :]

    def init_streaming_state(self, batch_size: int = 1, device=None, dtype=torch.float32):
        states = []
        for layer in self.layers:
            conv_ring = torch.zeros(
                batch_size, layer.d_conv - 1, layer.d_inner, device=device, dtype=dtype
            )
            h = torch.zeros(
                batch_size, layer.d_inner, layer.d_state, device=device, dtype=dtype
            )
            states.append((conv_ring, h))
        return states

    def step(self, token_40: torch.Tensor, states):
        """
        Streaming step on a single 40-bin log-mel token:
          token_40: (B, 40)
          states: list of (conv_ring, h) per layer
        Returns:
          logits: (B, 12)
          new_states: updated states
        """
        h = self.stem(token_40)
        new_states = []
        for layer, state in zip(self.layers, states):
            h, new_state = layer.step(h, state)
            new_states.append(new_state)

        logits = self.head(self.final_norm(h))
        return logits, new_states
