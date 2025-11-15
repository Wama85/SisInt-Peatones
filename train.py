import numpy as np
import os
import cv2
from load_data import load_dataset, extract_7x7, extract_conv_features
from mlp_model import *

# ========================== RUTAS ===============================

TRAIN_IMG = r"PEATONES/Train/JPEGImages"
TRAIN_XML = r"PEATONES/Train/Annotations"
TRAIN_TXT = r"PEATONES/Train/train.txt"

VAL_IMG = r"PEATONES/Val/JPEGImages"
VAL_XML = r"PEATONES/Val/Annotations"
VAL_TXT = r"PEATONES/Val/val.txt"

NEG_DIR = r"negativos"

# ========================== DEBUG ===============================

print("========== DEBUG ==========")
print("IMG:", TRAIN_IMG)
print("XML:", TRAIN_XML)
print("TXT:", TRAIN_TXT)
print("NEG:", NEG_DIR)

print("\nJPEGImages contenido:")
try:
    print(os.listdir(TRAIN_IMG)[:10])
except Exception as e:
    print("⚠ No se puede leer JPEGImages:", e)

print("\nAnnotations contenido:")
try:
    print(os.listdir(TRAIN_XML)[:10])
except Exception as e:
    print("⚠ No se puede leer Annotations:", e)

print("\nContenido train.txt:")
try:
    with open(TRAIN_TXT, "r", encoding="utf-8") as f:
        for i in range(10):
            linea = f.readline()
            if not linea:
                break
            print(linea.strip())
except Exception as e:
    print("⚠ No se puede leer train.txt:", e)

print("========== FIN DEBUG ==========\n")


# ========================== CARGA POSITIVOS ===============================

print("[INFO] Cargando dataset de PEATONES (positivos)...")
X_pos, Y_pos = load_dataset(TRAIN_IMG, TRAIN_XML, TRAIN_TXT, use_conv=True)

print("X_pos:", X_pos.shape)
print("Y_pos:", Y_pos.shape)

if X_pos.shape[1] == 0:
    print("❌ ERROR: No se pudo cargar ninguna imagen positiva.")
    exit()


# ========================== CARGA NEGATIVOS ===============================

def load_negatives(neg_dir):
    Xn, Yn = [], []

    if not os.path.isdir(neg_dir):
        print(f"⚠ Carpeta de negativos '{neg_dir}' no existe. No se cargarán negativos.")
        return np.empty((X_pos.shape[0], 0)), np.empty((1, 0))

    archivos = [f for f in os.listdir(neg_dir) if f.lower().endswith((".jpg", ".png", ".jpeg"))]

    if len(archivos) == 0:
        print(f"⚠ No hay imágenes negativas en '{neg_dir}'.")
        return np.empty((X_pos.shape[0], 0)), np.empty((1, 0))

    print(f"[INFO] Cargando NEGATIVOS desde '{neg_dir}' ({len(archivos)} archivos)...")

    for name in archivos:
        path = os.path.join(neg_dir, name)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            print(f"  - ⚠ No se pudo leer {path}")
            continue

        try:
            desc = extract_conv_features(img)
            Xn.append(desc)
            Yn.append(0)
        except Exception as e:
            print(f"  - ⚠ Error procesando {name}: {e}")
            continue

    if len(Xn) == 0:
        print("⚠ No se pudo generar ningún negativo válido.")
        return np.empty((X_pos.shape[0], 0)), np.empty((1, 0))

    Xn = np.array(Xn).T
    Yn = np.array(Yn).reshape(1, -1)

    print("[INFO] Negativos cargados:", Xn.shape[1])
    return Xn, Yn


X_neg, Y_neg = load_negatives(NEG_DIR)

print("X_neg:", X_neg.shape)
print("Y_neg:", Y_neg.shape)

# ========================== UNIR POSITIVOS + NEGATIVOS ===============================

if X_neg.shape[1] > 0:
    X_train = np.concatenate([X_pos, X_neg], axis=1)
    Y_train = np.concatenate([Y_pos, Y_neg], axis=1)
else:
    print("⚠ Entrenando SOLO con positivos (menos recomendable).")
    X_train, Y_train = X_pos, Y_pos

print("\n[INFO] Dataset final:")
print("X_train:", X_train.shape)
print("Y_train:", Y_train.shape)
print(f"  - Positivos: {int(Y_train.sum())}")
print(f"  - Negativos: {int(Y_train.shape[1] - Y_train.sum())}")

m = X_train.shape[1]
if m == 0:
    print("❌ ERROR: No hay datos para entrenar.")
    exit()


# ========================== ENTRENAMIENTO ==============================

n_x = X_train.shape[0]
n_h1 = 256
n_h2 = 128
n_y = 1

print(f"\n[INFO] Arquitectura de la red (3 capas):")
print(f"  - Capa de entrada: {n_x} neuronas")
print(f"  - Capa oculta 1: {n_h1} neuronas (ReLU)")
print(f"  - Capa oculta 2: {n_h2} neuronas (ReLU)")
print(f"  - Capa de salida: {n_y} neurona (Sigmoid)")

W1, b1, W2, b2, W3, b3 = initialize_params(n_x, n_h1, n_h2, n_y)

epochs = 20
lr = 0.01
batch_size = 32

print(f"\n[INFO] Hiperparámetros:")
print(f"  - Épocas: {epochs}")
print(f"  - Learning rate: {lr}")
print(f"  - Batch size: {batch_size}")

print("\n[INFO] Iniciando entrenamiento...\n")

for epoch in range(epochs):
    perm = np.random.permutation(m)
    X_shuffled = X_train[:, perm]
    Y_shuffled = Y_train[:, perm]

    for i in range(0, m, batch_size):
        Xb = X_shuffled[:, i:i+batch_size]
        Yb = Y_shuffled[:, i:i+batch_size]

        Z1, A1, Z2, A2, Z3, A3 = forward(Xb, W1, b1, W2, b2, W3, b3)
        dW1, db1, dW2, db2, dW3, db3 = backward(Xb, Yb, Z1, A1, Z2, A2, Z3, A3, W1, W2, W3)
        W1, b1, W2, b2, W3, b3 = update(W1, b1, W2, b2, W3, b3, 
                                         dW1, db1, dW2, db2, dW3, db3, lr)

    _, _, _, _, _, A3_full = forward(X_shuffled, W1, b1, W2, b2, W3, b3)
    cost = compute_cost(A3_full, Y_shuffled)
    predictions = (A3_full > 0.5).astype(int)
    accuracy = np.mean(predictions == Y_shuffled) * 100
    
    print(f"Época {epoch+1:2d}/{epochs} - Costo: {cost:.4f} - Accuracy: {accuracy:.2f}%")

# ========================== EVALUACIÓN FINAL ==============================

print("\n[INFO] Evaluación final:")
_, _, _, _, _, A3_train = forward(X_train, W1, b1, W2, b2, W3, b3)
predictions_train = (A3_train > 0.5).astype(int)
accuracy_train = np.mean(predictions_train == Y_train) * 100

true_positives = np.sum((predictions_train == 1) & (Y_train == 1))
false_positives = np.sum((predictions_train == 1) & (Y_train == 0))
false_negatives = np.sum((predictions_train == 0) & (Y_train == 1))

precision = true_positives / (true_positives + false_positives + 1e-8) * 100
recall = true_positives / (true_positives + false_negatives + 1e-8) * 100

print(f"  - Accuracy: {accuracy_train:.2f}%")
print(f"  - Precision: {precision:.2f}%")
print(f"  - Recall: {recall:.2f}%")

# ========================== GUARDAR MODELO ==============================

np.save("W1.npy", W1)
np.save("b1.npy", b1)
np.save("W2.npy", W2)
np.save("b2.npy", b2)
np.save("W3.npy", W3)
np.save("b3.npy", b3)

print("\n[OK] Modelo guardado correctamente")
print("\n✅ Entrenamiento completado exitosamente!")