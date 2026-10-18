# Pre-entrega 1: Pipeline de entrenamiento, validación y clasificador base

## Objetivo del checkpoint

Establecer la infraestructura técnica inicial de un sistema de aprendizaje profundo en PyTorch: un pipeline completo, reproducible y escalable que entrene un clasificador base sobre un dataset de clasificación binaria, evalúe su desempeño en datos no vistos y registre métricas por época.

## Estructura del repositorio

```
Pre Entregable 1/
├── data/
│   └── dataset_sintetico.npz      # Datasets de entrenamiento y validacion
├── src/
│   ├── train.py                   # Pipeline de entrenamiento y validacion
│   ├── base_classifier.pt         # Pesos del modelo entrenado
│   └── history.json               # Metricas por epoca (loss, accuracy, f1)
├── requirements.txt               # Dependencias del entorno
└── README.md                      # Documentacion del experimento
```

## Configuración del entorno

- **Dispositivo**: detección automática con prioridad `CUDA` > `MPS` (Apple Silicon) > `CPU`. Todos los tensores y el modelo se mueven al dispositivo con `.to(device)` para evitar inconsistencias CPU/GPU.
- **Reproducibilidad**: semillas fijas en `numpy`, `torch` y `torch.cuda` (`RANDOM_SEED = 42`).
- **Librerías esenciales**: `torch`, `torch.nn`, `torch.optim`, `numpy`, `scikit-learn`.
- **Versión de PyTorch utilizada**: `2.14.1+cpu`.

## Arquitectura base

`BaseClassifier` (`src/train.py`): MLP de dos capas con activación no lineal ReLU y Dropout:

- Entrada: 20 características
- Capa oculta: 16 neuronas + ReLU + Dropout (p=0.2)
- Salida: 2 logits (clasificación binaria) + `CrossEntropyLoss`

## Hiperparámetros

| Hiperparámetro | Valor |
|---|---|
| `learning_rate` | 1e-3 (optimizador Adam) |
| Épocas | 30 |
| Batch size | 32 |
| Dimensión oculta | 16 |
| Test size (validación) | 20 % |
| Semilla aleatoria | 42 |

## Ciclo de entrenamiento

1. **Forward pass**: `logits = model(inputs)`.
2. **Cálculo de pérdida**: `CrossEntropyLoss(logits, targets)`.
3. **Backward pass**: `optimizer.zero_grad()` → `loss.backward()` (motor `torch.autograd`) → `optimizer.step()` (actualización de pesos con Adam).
4. **Validación**: modo `model.eval()` con `@torch.no_grad()`; se calculan pérdida, Accuracy y F1-score en el conjunto de validación separado.

## Resultados del experimento

| Época | train_loss | train_acc | val_loss | val_acc | val_f1 |
|---|---|---|---|---|---|
| 1 | 0.6270 | 0.6719 | 0.5909 | 0.7450 | 0.7692 |
| 15 | 0.2423 | 0.9100 | 0.2382 | 0.9025 | 0.9023 |
| 30 | 0.1921 | 0.9350 | 0.2316 | 0.9200 | 0.9208 |

### Interpretación de la curva de pérdida

La pérdida de entrenamiento disminuye de forma consistente durante las 30 épocas (de 0.63 a ~0.19), y la pérdida de validación baja acompañándola (de 0.59 a ~0.23) sin abrirse una brecha significativa entre ambas curvas. Esto indica que el modelo aprende el patrón de los datos, converge y no presenta overfitting temprano en este checkpoint. La Accuracy de validación se estabiliza alrededor de 0.91-0.92.

## Cómo ejecutar

```bash
# 1. Crear entorno e instalar dependencias
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

# 2. Ejecutar el pipeline
python src/train.py
```

El script genera automáticamente el dataset sintético, lo guarda en `data/`, entrena el modelo y guarda los pesos en `src/base_classifier.pt` junto con el historial de métricas en `src/history.json`.

## Criterios de cumplimiento

- [x] Ejecución sin errores durante todas las épocas
- [x] Uso correcto de `torch.autograd` mediante `.backward()` con `zero_grad()` antes de cada paso
- [x] Consistencia de dispositivos mediante `.to(device)`
- [x] Separación entrenamiento/validación para detección temprana de overfitting
- [x] Registro de pérdida y Accuracy (y F1-score) por época
- [x] Semillas fijadas para reproducibilidad
