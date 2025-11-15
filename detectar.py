import cv2
import numpy as np
from mlp_model import forward
from load_data import extract_7x7

# Cargar modelo entrenado
W1 = np.load("W1.npy")
b1 = np.load("b1.npy")
W2 = np.load("W2.npy")
b2 = np.load("b2.npy")

# Ruta de la imagen
path = "foto.jpg"
img = cv2.imread(path)

if img is None:
    print(f"[ERROR] No se pudo cargar la imagen '{path}'. Verifica la ruta.")
    exit()

# Procesar imagen
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Redimensionar (si quieres 64×64 o 32×32, aquí se ajusta)
img_gray_resized = cv2.resize(img_gray, (64,64))

# Extraer características con kernel 7x7
desc = extract_7x7(img_gray_resized)
X = desc.reshape(-1,1)

# Evaluar en la red
_, _, _, A2 = forward(X, W1, b1, W2, b2)

print("Probabilidad de rostro:", float(A2))
