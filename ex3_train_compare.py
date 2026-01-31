import time
import math
from typing import Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from codecarbon import EmissionsTracker


def make_data(n_samples: int = 20000, n_features: int = 100, n_classes: int = 10) -> Tuple[torch.Tensor, torch.Tensor]:
    X = torch.randn(n_samples, n_features)
    W = torch.randn(n_features, n_classes) * 0.1
    y_logits = X @ W + 0.01 * torch.randn(n_samples, n_classes)
    y = y_logits.argmax(dim=1)
    return X, y


class SmallMLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x):
        return self.net(x)


def train_one(device: torch.device, epochs: int = 3) -> Tuple[float, float]:
    X, y = make_data()
    ds = TensorDataset(X, y)
    dl = DataLoader(ds, batch_size=256, shuffle=True, pin_memory=(device.type == 'cuda'))

    model = SmallMLP(in_dim=X.shape[1], hidden=256, out_dim=int(y.max().item()) + 1).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    t0 = time.perf_counter()
    for _ in range(epochs):
        for xb, yb in dl:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            out = model(xb)
            loss = loss_fn(out, yb)
            loss.backward()
            opt.step()
    torch.cuda.synchronize() if device.type == 'cuda' else None
    return loss.item(), time.perf_counter() - t0


def run_with_emissions(device: torch.device, project: str, run_id: str) -> Tuple[float, float, float]:
    tracker = EmissionsTracker(project_name=project, experiment_id=run_id, measure_power_secs=1)
    tracker.start()
    try:
        final_loss, duration = train_one(device)
    finally:
        emissions = tracker.stop() or 0.0
    return final_loss, duration, emissions


def main():
    project = "ex3_train_compare"
    cpu_device = torch.device('cpu')
    gpu_available = torch.cuda.is_available()
    gpu_device = torch.device('cuda') if gpu_available else None

    print(f"GPU available: {gpu_available}")

    loss_cpu, t_cpu, e_cpu = run_with_emissions(cpu_device, project, 'cpu')
    print(f"CPU -> time: {t_cpu:.2f}s, loss: {loss_cpu:.4f}, emissions: {e_cpu:.8f} kg CO2e")

    if gpu_device:
        time.sleep(0.5)
        loss_gpu, t_gpu, e_gpu = run_with_emissions(gpu_device, project, 'gpu')
        print(f"GPU -> time: {t_gpu:.2f}s, loss: {loss_gpu:.4f}, emissions: {e_gpu:.8f} kg CO2e")

        if e_gpu < e_cpu:
            print(f"Result: GPU emitted ~{e_cpu / max(e_gpu, 1e-12):.1f}x LESS CO2e than CPU.")
        else:
            print(f"Result: GPU emitted ~{e_gpu / max(e_cpu, 1e-12):.1f}x MORE CO2e than CPU.")
    else:
        print("GPU not available; only CPU run measured.")

    print("Note: For short/small workloads, GPU may not be efficient due to overheads; for larger batches/epochs, GPU often wins in time and sometimes in emissions.")


if __name__ == "__main__":
    main()
