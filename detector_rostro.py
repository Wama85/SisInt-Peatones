import cv2
import numpy as np
from mlp_model import forward
from load_data import extract_7x7   # <--- IMPORTANTE

# cargar modelo entrenado
W1 = np.load("W1.npy")
b1 = np.load("b1.npy")
W2 = np.load("W2.npy")
b2 = np.load("b2.npy")

# imagen a procesar
ruta_imagen = "foto.jpg"
img = cv2.imread(ruta_imagen)

if img is None:
    print("[ERROR] No se pudo cargar la imagen.")
    exit()

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Sliding window: ventana grande
win_size = 64
step = 16

detecciones = []
h, w = gray.shape

print("[INFO] Escaneando imagen...")

for y in range(0, h - win_size, step):
    for x in range(0, w - win_size, step):
        # recortar ventana de 64x64
        ventana = gray[y:y+win_size, x:x+win_size]

        # EXTRAER CARACTERÍSTICAS CON KERNEL 7x7
        desc = extract_7x7(ventana)
        X = desc.reshape(-1, 1)

        # clasificar
        _, _, _, A2 = forward(X, W1, b1, W2, b2)
        prob = float(A2)

        if prob > 0.90:
            detecciones.append((x, y, x+win_size, y+win_size, prob))


# dibujar detecciones
for (x1, y1, x2, y2, prob) in detecciones:
    cv2.rectangle(img, (x1, y1), (x2, y2), (0,255,0), 2)
    cv2.putText(img, f"{prob:.2f}", (x1, y1-5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

cv2.imshow("Detecciones", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

print(f"[OK] Se detectaron {len(detecciones)} posibles rostros.")
