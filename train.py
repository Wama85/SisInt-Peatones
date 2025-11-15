import numpy as np
import os
from load_data import load_dataset, extract_7x7
from mlp_model import *

# ========================== RUTAS ===============================

TRAIN_IMG = r"PEATONES/Train/JPEGImages"
TRAIN_XML = r"PEATONES/Train/Annotations"
TRAIN_TXT = r"PEATONES/Train/train.txt"

VAL_IMG = r"PEATONES/Val/JPEGImages"
VAL_XML = r"PEATONES/Val/Annotations"
VAL_TXT = r"PEATONES/Val/val.txt"

NEG_DIR = r"negativos"   # carpeta donde guardas imágenes negativas (NO rostros)

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

print("[INFO] Cargando dataset de ROSTROS (positivos)...")
X_pos, Y_pos = load_dataset(TRAIN_IMG, TRAIN_XML, TRAIN_TXT)

print("X_pos:", X_pos.shape)   # (n_features, m_pos)
print("Y_pos:", Y_pos.shape)   # (1, m_pos)

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
        img = cv2.imread(path)

        if img is None:
            print(f"  - ⚠ No se pudo leer {path}")
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Usamos el mismo descriptor 7x7 que para los rostros
        desc = extract_7x7(gray)
        Xn.append(desc)
        Yn.append(0)   # etiqueta 0 = NO rostro

    if len(Xn) == 0:
        print("⚠ No se pudo generar ningún negativo válido.")
        return np.empty((X_pos.shape[0], 0)), np.empty((1, 0))

    Xn = np.array(Xn).T   # (n_features, m_neg)
    Yn = np.array(Yn).reshape(1, -1)

    print("[INFO] Negativos cargados:", Xn.shape[1])
    return Xn, Yn


# cargar negativos
import cv2  # lo importamos aquí para usarlo en load_negatives
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
print("X_train:", X_train.shape)   # (n_features, m_total)
print("Y_train:", Y_train.shape)   # (1, m_total)

m = X_train.shape[1]
if m == 0:
    print("❌ ERROR: No hay datos para entrenar.")
    exit()


# ========================== ENTRENAMIENTO ==============================

n_x = X_train.shape[0]  # número de características (debería ser 49 si es 7x7)
n_h = 256
n_y = 1

W1, b1, W2, b2 = initialize_params(n_x, n_h, n_y)

epochs = 20
lr = 0.01
batch_size = 32

print("\n[INFO] Iniciando entrenamiento...\n")

for epoch in range(epochs):
    # Barajar datos
    perm = np.random.permutation(m)
    X_train = X_train[:, perm]
    Y_train = Y_train[:, perm]

    for i in range(0, m, batch_size):
        Xb = X_train[:, i:i+batch_size]
        Yb = Y_train[:, i:i+batch_size]

        Z1, A1, Z2, A2 = forward(Xb, W1, b1, W2, b2)
        dW1, db1, dW2, db2 = backward(Xb, Yb, Z1, A1, Z2, A2, W1, W2)
        W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, lr)

    _, _, _, A2_full = forward(X_train, W1, b1, W2, b2)
    cost = compute_cost(A2_full, Y_train, W1, W2)
    print(f"Época {epoch+1}/{epochs} - Costo: {cost:.4f}")

# ========================== GUARDAR MODELO ==============================

np.save("W1.npy", W1)
np.save("b1.npy", b1)
np.save("W2.npy", W2)
np.save("b2.npy", b2)

print("\n[OK] Modelo guardado correctamente")
