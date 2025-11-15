import cv2
import numpy as np
from mlp_model import forward
from load_data import extract_7x7   # <--- IMPORTANTE

# ==========================
# CARGAR MODELO ENTRENADO
# ==========================
W1 = np.load("W1.npy")
b1 = np.load("b1.npy")
W2 = np.load("W2.npy")
b2 = np.load("b2.npy")


# ==========================
# NON-MAX SUPPRESSION (NMS)
# ==========================
def non_max_suppression(boxes, overlapThresh=0.3):
    if len(boxes) == 0:
        return []

    boxes = np.array(boxes)

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    scores = boxes[:, 4]

    area = (x2 - x1) * (y2 - y1)
    idxs = np.argsort(scores)[::-1]

    pick = []

    while len(idxs) > 0:
        i = idxs[0]
        pick.append(i)

        xx1 = np.maximum(x1[i], x1[idxs[1:]])
        yy1 = np.maximum(y1[i], y1[idxs[1:]])
        xx2 = np.minimum(x2[i], x2[idxs[1:]])
        yy2 = np.minimum(y2[i], y2[idxs[1:]])

        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)
        inter = w * h

        overlap = inter / (area[idxs[1:]] + 1e-5)

        idxs = np.delete(
            idxs,
            np.concatenate(([0], np.where(overlap > overlapThresh)[0] + 1))
        )

    return boxes[pick].astype(int)


# ==========================
# CARGAR IMAGEN
# ==========================
ruta_imagen = "foto.jpg"
img = cv2.imread(ruta_imagen)

if img is None:
    print("[ERROR] No se pudo cargar la imagen.")
    exit()

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
h, w = gray.shape

print("[INFO] Escaneando imagen...")


# ==========================
# SLIDING WINDOW 128x128
# ==========================
win_size = 128
step = 32

raw_detections = []

for y in range(0, h - win_size, step):
    for x in range(0, w - win_size, step):

        x1, y1 = x, y
        x2, y2 = x + win_size, y + win_size

        ventana = gray[y1:y2, x1:x2]

        # ======= EXTRAER DESCRIPTORES 7×7 =======
        desc = extract_7x7(ventana)
        X = desc.reshape(-1, 1)

        # ======= CLASIFICAR =======
        _, _, _, A2 = forward(X, W1, b1, W2, b2)
        prob = float(A2.squeeze())

        # ==========================================
        # 💥 FILTROS PARA ELIMINAR FALSOS POSITIVOS
        # ==========================================

        # 1. Evitar detectar zonas muy bajas (piernas)
        if y1 > h * 0.55:
            continue

        # 2. Tamaño mínimo para considerar rostro
        width = x2 - x1
        height = y2 - y1
        if width < 80 or height < 80:
            continue

        # 3. Proporción del rostro (aprox cuadrado)
        ratio = width / height
        if ratio < 0.70 or ratio > 1.30:
            continue

        # 4. Probabilidad mínima
        if prob < 0.90:
            continue

        raw_detections.append([x1, y1, x2, y2, prob])

print(f"[INFO] Detecciones brutas: {len(raw_detections)}")


# ==========================
# APLICAR NMS
# ==========================
final_boxes = non_max_suppression(raw_detections, overlapThresh=0.35)
print(f"[OK] Rostros finales detectados: {len(final_boxes)}")


# ==========================
# DIBUJAR RESULTADOS
# ==========================
for (x1, y1, x2, y2, score) in final_boxes:
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(img, f"{score:.3f}", (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

cv2.imshow("DETECCIÓN 7x7 AJUSTADA", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
