"""The learned-A ablation starts out as exactly the C1 model, and leaves the default model untouched."""
import torch

from train.kws.model import MuMambaKWS


def _models():
    torch.manual_seed(0)
    c1 = MuMambaKWS()
    learned = MuMambaKWS(learned_a=True)
    missing, unexpected = learned.load_state_dict(c1.state_dict(), strict=False)
    return c1, learned, missing, unexpected


def test_default_state_dict_has_no_a_log():
    assert not any("A_log" in k for k in MuMambaKWS().state_dict())


def test_learned_a_adds_only_a_log_and_matches_c1_at_init():
    c1, learned, missing, unexpected = _models()
    assert not unexpected and all(k.endswith("A_log") for k in missing) and len(missing) == 2
    x = torch.randn(3, 40, 40) * 3.0
    with torch.no_grad():
        assert torch.allclose(c1.forward_seq(x), learned.forward_seq(x), atol=1e-5, rtol=1e-5)


def test_learned_a_step_matches_forward_seq():
    _, learned, _, _ = _models()
    x = torch.randn(2, 12, 40)
    with torch.no_grad():
        ref = learned.forward_seq(x)
        states = learned.init_streaming_state(batch_size=2)
        for t in range(12):
            out, states = learned.step(x[:, t], states)
            assert torch.allclose(out, ref[:, t], atol=1e-5, rtol=1e-5)


def test_a_log_receives_gradient():
    _, learned, _, _ = _models()
    learned.forward_seq(torch.randn(2, 10, 40)).sum().backward()
    assert all(l.A_log.grad is not None and l.A_log.grad.abs().sum() > 0 for l in learned.layers)
