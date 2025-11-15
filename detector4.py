import cv2
import numpy as np
from mlp_model import forward
from load_data import extract_conv_features

print("[INFO] Cargando modelo...")
W1 = np.load("W1.npy")
b1 = np.load("b1.npy")
W2 = np.load("W2.npy")
b2 = np.load("b2.npy")
W3 = np.load("W3.npy")
b3 = np.load("b3.npy")
print("[OK] Modelo cargado")

def non_max_suppression(boxes, overlapThresh=0.3):
    if len(boxes) == 0:
        return []
    boxes = np.array(boxes)
    x1, y1, x2, y2, scores = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3], boxes[:, 4]
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
        overlap = (w * h) / (area[idxs[1:]] + 1e-5)
        idxs = np.delete(idxs, np.concatenate(([0], np.where(overlap > overlapThresh)[0] + 1)))
    
    return boxes[pick].astype(int)

img = cv2.imread("foto.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
h, w = gray.shape

print(f"[INFO] Escaneando imagen {w}x{h}...")

detections = []
win_size = 128
step = 32

for y in range(0, h - win_size, step):
    for x in range(0, w - win_size, step):
        ventana = gray[y:y+win_size, x:x+win_size]
        
        try:
            desc = extract_conv_features(ventana)
            X = desc.reshape(-1, 1)
        except:
            continue
        
        _, _, _, _, _, A3 = forward(X, W1, b1, W2, b2, W3, b3)
        prob = float(A3.squeeze())
        
        if prob > 0.9:
            detections.append([x, y, x+win_size, y+win_size, prob])

print(f"[INFO] Detecciones brutas: {len(detections)}")

final_boxes = non_max_suppression(detections, overlapThresh=0.2)

print(f"[OK] Caras detectadas: {len(final_boxes)}")

for idx, (x1, y1, x2, y2, score) in enumerate(final_boxes, 1):
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
    cv2.putText(img, f"CARA {idx}: {score:.2f}", (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

cv2.imwrite("resultado_deteccion.jpg", img)
cv2.imshow("Detección de Caras", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

print("[OK] Proceso completado")