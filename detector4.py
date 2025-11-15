import cv2
import numpy as np
from mlp_model import forward
from load_data import extract_7x7

# ==========================
# CARGAR MODELO ENTRENADO (3 CAPAS)
# ==========================
print("[INFO] Cargando modelo...")
try:
    W1 = np.load("W1.npy")
    b1 = np.load("b1.npy")
    W2 = np.load("W2.npy")
    b2 = np.load("b2.npy")
    W3 = np.load("W3.npy")
    b3 = np.load("b3.npy")
    print("[OK] Modelo cargado correctamente")
except FileNotFoundError as e:
    print(f"[ERROR] No se encontró el archivo del modelo: {e}")
    exit()

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
print(f"[INFO] Cargando imagen: {ruta_imagen}")
img = cv2.imread(ruta_imagen)

if img is None:
    print(f"[ERROR] No se pudo cargar la imagen: {ruta_imagen}")
    exit()

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
h, w = gray.shape
print(f"[INFO] Dimensiones: {w}x{h}")

# ==========================
# SLIDING WINDOW
# ==========================
win_size = 128
step = 16  # Paso más pequeño para mejor cobertura

print(f"[INFO] Escaneando imagen...")
raw_detections = []

for y in range(0, h - win_size, step):
    for x in range(0, w - win_size, step):
        x1, y1 = x, y
        x2, y2 = x + win_size, y + win_size
        
        ventana = gray[y1:y2, x1:x2]
        
        # Extraer descriptores
        try:
            desc = extract_7x7(ventana)
            X = desc.reshape(-1, 1)
        except:
            continue
        
        # Clasificar
        _, _, _, _, _, A3 = forward(X, W1, b1, W2, b2, W3, b3)
        prob = float(A3.squeeze())
        
        # ==========================================
        # FILTROS MUY ESTRICTOS PARA SOLO CARAS REALES
        # ==========================================
        
        # 1. Umbral de confianza MUY ALTO
        if prob < 0.999:  # Solo las detecciones más seguras
            continue
        
        # 2. Proporción de rostro (casi cuadrado)
        ratio = (x2 - x1) / (y2 - y1)
        if ratio < 0.85 or ratio > 1.15:
            continue
        
        # 3. Tamaño mínimo
        if (x2 - x1) < 80:
            continue
        
        raw_detections.append([x1, y1, x2, y2, prob])

print(f"[INFO] Detecciones brutas: {len(raw_detections)}")

# ==========================
# APLICAR NMS AGRESIVO
# ==========================
if len(raw_detections) == 0:
    print("[WARNING] No se detectaron caras")
    final_boxes = []
else:
    # NMS muy agresivo para eliminar solapamientos
    final_boxes = non_max_suppression(raw_detections, overlapThresh=0.2)
    print(f"[OK] Caras detectadas: {len(final_boxes)}")

# ==========================
# DIBUJAR RESULTADOS
# ==========================
for (x1, y1, x2, y2, score) in final_boxes:
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
    cv2.putText(img, f"CARA: {score:.3f}", (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

# ==========================
# MOSTRAR Y GUARDAR
# ==========================
cv2.imwrite("resultado_deteccion.jpg", img)
print(f"[OK] Resultado guardado")

cv2.imshow("Deteccion de Caras", img)
print("[INFO] Presiona cualquier tecla para cerrar...")
cv2.waitKey(0)
cv2.destroyAllWindows()