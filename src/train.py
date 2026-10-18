"""Pipeline base de entrenamiento y validacion en PyTorch.

Checkpoint Pre-entrega 1: Clasificador base sobre datos sinteticos de clasificacion.

Ejecucion:
    python src/train.py
"""

import os
import json
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score

# ---------------------------------------------------------------------------
# 1. Configuracion del entorno y reproducibilidad
# ---------------------------------------------------------------------------
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)

# Seleccion automatica del dispositivo: CUDA > MPS (Apple Silicon) > CPU
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")

print(f"Dispositivo seleccionado: {DEVICE}")
print(f"PyTorch version: {torch.__version__}")

# Hiperparametros
LEARNING_RATE = 1e-3
EPOCHS = 30
BATCH_SIZE = 32
HIDDEN_DIM = 16
TEST_SIZE = 0.2

# ---------------------------------------------------------------------------
# 2. Dataset sintético de clasificación binaria
# ---------------------------------------------------------------------------
N_SAMPLES = 2000
N_FEATURES = 20
N_CLASSES = 2

X, y = make_classification(
    n_samples=N_SAMPLES,
    n_features=N_FEATURES,
    n_informative=10,
    n_redundant=5,
    n_classes=N_CLASSES,
    random_state=RANDOM_SEED,
    class_sep=1.0,
)

# Division entrenamiento / validacion (datos no vistos durante el entrenamiento)
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y
)

# Estandarizacion (ajustada solo sobre entrenamiento para evitar fuga de informacion)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train).astype(np.float32)
X_val = scaler.transform(X_val).astype(np.float32)

y_train = y_train.astype(np.int64)
y_val = y_val.astype(np.int64)

# Tensores y DataLoaders movidos al dispositivo detectado
train_dataset = TensorDataset(
    torch.from_numpy(X_train), torch.from_numpy(y_train)
)
val_dataset = TensorDataset(
    torch.from_numpy(X_val), torch.from_numpy(y_val)
)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Guardar dataset para trazabilidad
os.makedirs("data", exist_ok=True)
np.savez(
    "data/dataset_sintetico.npz",
    X_train=X_train,
    X_val=X_val,
    y_train=y_train,
    y_val=y_val,
)
print(f"Datos de entrenamiento: {X_train.shape[0]} muestras")
print(f"Datos de validacion:    {X_val.shape[0]} muestras")

# ---------------------------------------------------------------------------
# 3. Arquitectura base: MLP de pocas capas con activaciones no lineales (ReLU)
# ---------------------------------------------------------------------------
class BaseClassifier(nn.Module):
    """Clasificador base: capa lineal -> ReLU -> Dropout -> capa lineal de salida."""

    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


model = BaseClassifier(
    input_dim=N_FEATURES, hidden_dim=HIDDEN_DIM, num_classes=N_CLASSES
).to(DEVICE)

# Funcion de perdida y optimizador Adam (motor de autograd a traves de .backward())
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

print(f"Modelo:\n{model}")
print(f"Learning rate: {LEARNING_RATE} | Epochas: {EPOCHS} | Batch size: {BATCH_SIZE}")

# ---------------------------------------------------------------------------
# 4. Funciones de entrenamiento y evaluacion
# ---------------------------------------------------------------------------
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    all_preds = []
    all_targets = []

    for inputs, targets in loader:
        # Consistencia de dispositivos: todos los tensores en el mismo device
        inputs = inputs.to(device)
        targets = targets.to(device)

        # Fase forward
        logits = model(inputs)
        loss = criterion(logits, targets)

        # Fase backward: reinicio de gradientes acumulados (zero_grad)
        optimizer.zero_grad()
        loss.backward()  # torch.autograd calcula gradientes
        optimizer.step()  # actualizacion de pesos con Adam

        running_loss += loss.item() * inputs.size(0)
        preds = logits.argmax(dim=1)
        all_preds.extend(preds.detach().cpu().numpy())
        all_targets.extend(targets.detach().cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = accuracy_score(all_targets, all_preds)
    return epoch_loss, epoch_acc


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_targets = []

    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        logits = model(inputs)
        loss = criterion(logits, targets)

        running_loss += loss.item() * inputs.size(0)
        preds = logits.argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_targets.extend(targets.cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = accuracy_score(all_targets, all_preds)
    epoch_f1 = f1_score(all_targets, all_preds, average="binary")
    return epoch_loss, epoch_acc, epoch_f1


# ---------------------------------------------------------------------------
# 5. Ciclo de entrenamiento con tracking de metricas por epoca
# ---------------------------------------------------------------------------
history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "val_f1": []}

for epoch in range(1, EPOCHS + 1):
    train_loss, train_acc = train_one_epoch(
        model, train_loader, criterion, optimizer, DEVICE
    )
    val_loss, val_acc, val_f1 = evaluate(model, val_loader, criterion, DEVICE)

    history["train_loss"].append(train_loss)
    history["train_acc"].append(train_acc)
    history["val_loss"].append(val_loss)
    history["val_acc"].append(val_acc)
    history["val_f1"].append(val_f1)

    print(
        f"Epoch {epoch:02d}/{EPOCHS} | "
        f"train_loss: {train_loss:.4f} train_acc: {train_acc:.4f} | "
        f"val_loss: {val_loss:.4f} val_acc: {val_acc:.4f} val_f1: {val_f1:.4f}"
    )

# ---------------------------------------------------------------------------
# 6. Guardado de metricas y modelo
# ---------------------------------------------------------------------------
os.makedirs("src", exist_ok=True)
torch.save(model.state_dict(), "src/base_classifier.pt")
with open("src/history.json", "w", encoding="utf-8") as f:
    json.dump(history, f, indent=2)

print("\nEntrenamiento finalizado.")
print(f"Modelo guardado en src/base_classifier.pt")
print(f"Historial guardado en src/history.json")
