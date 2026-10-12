# train.py
import time
import torch
import torch.nn as nn
import tracemalloc
from torch.utils.tensorboard import SummaryWriter


def load_data():
    time.sleep(0.1)  # Simula leitura de I/O
    return torch.randn(32, 784)


def forward_pass(model, data):
    return model(data)


def train_loop(writer):
    model = nn.Sequential(nn.Linear(784, 128), nn.ReLU(), nn.Linear(128, 10))

    for step in range(10):
        batch = load_data()
        output = forward_pass(model, batch)
        loss = output.sum()
        loss.backward()

        # 1. PONTO DE INTERRUPÇÃO CONDICIONAL
        # Por exemplo: Interrompe a execução para inspeção no passo 5
        if step == 5:
            print(f"\n[DEBUG] Interrupção no passo {step} para inspeção!")
            breakpoint()  # <--- Abre o prompt do depurador interativo (pdb)

        writer.add_scalar("Loss/train", loss.item(), global_step=step)
        print(f"Step {step} | Loss: {loss.item():.4f}")


if __name__ == "__main__":
    writer = SummaryWriter(log_dir="runs/treino_experimento_1")
    tracemalloc.start()

    train_loop(writer)

    writer.close()
    tracemalloc.stop()