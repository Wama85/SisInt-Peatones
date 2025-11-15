import cv2
import numpy as np
from mlp_model import forward
from load_data import extract_7x7

# cargar modelo entrenado
W1 = np.load("W1.npy")
b1 = np.load("b1.npy")
W2 = np.load("W2.npy")
b2 = np.load("b2.npy")

# ------- NMS -------
def non_max_suppression(boxes, thresh=0.4):
    if len(boxes) == 0:
        return []

    boxes = np.array(boxes, dtype=float)

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
            np.concatenate(([0], np.where(overlap > thresh)[0] + 1))
        )

    return boxes[pick].astype(int)

# ------- AGRUPAR DETECCIONES -------
def agrupar_por_distancia(boxes, max_dist=100):
    boxes = np.array(boxes)
    usados = set()
    grupos = []

    for i in range(len(boxes)):
        if i in usados:
            continue

        x1, y1, x2, y2, p = boxes[i]
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2

        grupo = [i]
        usados.add(i)

        for j in range(i + 1, len(boxes)):
            if j in usados:
                continue

            x1b, y1b, x2b, y2b, pb = boxes[j]
            cxb = (x1b + x2b) / 2
            cyb = (y1b + y2b) / 2

            dist = np.sqrt((cx - cxb)**2 + (cy - cyb)**2)

            if dist < max_dist:
                grupo.append(j)
                usados.add(j)

        grupos.append(grupo)

    # Fusionar grupos
    finales = []
    for g in grupos:
        sub = boxes[g]
        x1 = int(np.mean(sub[:,0]))
        y1 = int(np.mean(sub[:,1]))
        x2 = int(np.mean(sub[:,2]))
        y2 = int(np.mean(sub[:,3]))
        p  = float(np.max(sub[:,4]))
        finales.append([x1,y1,x2,y2,p])

    return finales

# ------- CARGAR IMAGEN -------
img = cv2.imread("foto.jpg")
if img is None:
    print("ERROR cargando imagen")
    exit()

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

win_size = 64
step = 16
raw_det = []

h, w = gray.shape

print("[INFO] Escaneando imagen...")

for y in range(0, h - win_size, step):
    for x in range(0, w - win_size, step):

        ventana = gray[y:y+win_size, x:x+win_size]

        desc = extract_7x7(ventana)
        X = desc.reshape(-1,1)

        _,_,_,A2 = forward(X, W1, b1, W2, b2)
        prob = float(A2.squeeze())

        # FILTRO más estricto
        if prob > 0.97:
            raw_det.append([x, y, x+win_size, y+win_size, prob])

print("[INFO] Detecciones brutas:", len(raw_det))

# NMS agresivo
nms = non_max_suppression(raw_det, 0.35)

# Agrupación final
final = agrupar_por_distancia(nms, max_dist=85)

print("[OK] Rostros finales detectados:", len(final))

# ------- DIBUJAR -------
for (x1,y1,x2,y2,p) in final:
    cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
    cv2.putText(img,f"{p:.2f}",(x1,y1-5),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,0),2)

cv2.imshow("Detecciones 7x7 PRO", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
