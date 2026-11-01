"""Standalone GPU diagnostic for KWS-Mu-Mamba.

Run from the repo root after a reboot/driver fix:
    python tools/check_gpu.py
"""
import os
import platform
import shutil
import subprocess
import sys

# Ensure repo root is on sys.path so `python tools/check_gpu.py` works.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


def main() -> int:
    import torch

    print(f"kernel: {platform.release()}")

    nvidia_version_path = "/proc/driver/nvidia/version"
    if os.path.exists(nvidia_version_path):
        try:
            with open(nvidia_version_path) as f:
                first_line = f.readline().strip()
            print(f"{nvidia_version_path}: exists")
            print(f"nvidia version: {first_line}")
        except Exception as e:
            print(f"{nvidia_version_path}: exists (unreadable: {e})")
    else:
        print(f"{nvidia_version_path}: missing")

    lspci_driver = None
    if shutil.which("lspci") is not None:
        try:
            out = subprocess.run(
                ["lspci", "-k"], capture_output=True, text=True, timeout=10
            ).stdout
            # Find NVIDIA VGA/3D block and its "Kernel driver in use" line.
            lines = out.splitlines()
            for i, line in enumerate(lines):
                if "NVIDIA" in line:
                    for j in range(i, min(i + 5, len(lines))):
                        if "Kernel driver in use:" in lines[j]:
                            lspci_driver = lines[j].split(
                                "Kernel driver in use:"
                            )[1].strip()
                            break
                    if lspci_driver:
                        break
            if lspci_driver:
                print(f"lspci driver: {lspci_driver}")
            else:
                print("lspci driver: not found")
        except Exception as e:
            print(f"lspci driver: unknown (lspci failed: {e})")
    else:
        print("lspci driver: unknown (lspci not available)")

    print(f"torch version: {torch.__version__}")
    print(f"torch.version.cuda: {torch.version.cuda}")
    cuda_available = torch.cuda.is_available()
    print(f"torch.cuda.is_available(): {cuda_available}")
    if cuda_available:
        print(f"device name: {torch.cuda.get_device_name(0)}")
        total = torch.cuda.get_device_properties(0).total_memory
        print(f"total VRAM: {total / (1024 ** 2):.0f} MiB")

    if cuda_available:
        from train.kws.model import MuMambaKWS
        from train.kws.train import sequence_criterion

        torch.cuda.reset_peak_memory_stats()
        model = MuMambaKWS().to("cuda")
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        model.train()
        for step in range(3):
            feats = torch.randn(8, 200, 40, device="cuda")
            targets = torch.randint(0, 12, (8, 200), device="cuda")
            opt.zero_grad()
            logits = model.forward_seq(feats)
            loss = sequence_criterion(logits, targets)
            assert torch.isfinite(loss).all(), f"non-finite loss: {loss}"
            loss.backward()
            opt.step()
        peak_mib = torch.cuda.max_memory_allocated() / (1024 ** 2)
        print(f"peak VRAM: {peak_mib:.0f} MiB")
        print("GPU TRAINING OK")
        return 0

    # CUDA not available: report specific reason and fix.
    print("GPU NOT AVAILABLE")
    if torch.version.cuda is None:
        print("reason: torch built without CUDA (CPU-only build)")
        print("fix: pip install --no-cache-dir \"torch==2.14.1+cu130\" "
              "--index-url https://download.pytorch.org/whl/cu130")
    elif not os.path.exists(nvidia_version_path):
        print("reason: nvidia kernel module not loaded")
        print("fix: sudo pacman -Syu then reboot")
    elif lspci_driver == "nouveau":
        print("reason: nouveau bound to the card")
        print("fix: install/enable the NVIDIA driver (sudo pacman -Syu), "
              "blacklist nouveau, then reboot")
    else:
        print("reason: CUDA unavailable (driver may not be loaded)")
        print("fix: sudo pacman -Syu then reboot")
    return 1


if __name__ == "__main__":
    sys.exit(main())
