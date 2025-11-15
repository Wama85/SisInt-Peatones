import cv2
import numpy as np
import os
import xml.etree.ElementTree as ET

# ========================== CONFIGURACIÓN ===============================

TRAIN_IMG = r"PEATONES/Train/JPEGImages"
TRAIN_XML = r"PEATONES/Train/Annotations"
NEG_OUTPUT = r"negativos"

OBJETIVO_NEGATIVOS = 1000

os.makedirs(NEG_OUTPUT, exist_ok=True)

# ========================== FUNCIONES ===============================

def extraer_bboxes_xml(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    boxes = []
    
    for obj in root.findall('object'):
        bbox = obj.find('bndbox')
        xmin = int(bbox.find('xmin').text)
        ymin = int(bbox.find('ymin').text)
        xmax = int(bbox.find('xmax').text)
        ymax = int(bbox.find('ymax').text)
        boxes.append((xmin, ymin, xmax, ymax))
    
    return boxes

def hay_solapamiento(x, y, w, h, bboxes, margen=25):
    for (bx1, by1, bx2, by2) in bboxes:
        bx1 -= margen
        by1 -= margen
        bx2 += margen
        by2 += margen
        
        ix1 = max(x, bx1)
        iy1 = max(y, by1)
        ix2 = min(x + w, bx2)
        iy2 = min(y + h, by2)
        
        if ix1 < ix2 and iy1 < iy2:
            return True
    
    return False

# ========================== GENERACIÓN ===============================

print("=" * 70)
print("GENERADOR DE EJEMPLOS NEGATIVOS")
print("=" * 70)
print("\n[INFO] Extrayendo regiones SIN peatones...\n")

contador = 0
imagenes = [f for f in os.listdir(TRAIN_IMG) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]

for img_name in imagenes:
    if contador >= OBJETIVO_NEGATIVOS:
        break
    
    base_name = os.path.splitext(img_name)[0]
    img_path = os.path.join(TRAIN_IMG, img_name)
    xml_path = os.path.join(TRAIN_XML, base_name + '.xml')
    
    if not os.path.exists(xml_path):
        continue
    
    img = cv2.imread(img_path)
    if img is None:
        continue
    
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    bboxes = extraer_bboxes_xml(xml_path)
    
    intentos = 0
    while intentos < 30 and contador < OBJETIVO_NEGATIVOS:
        intentos += 1
        
        size = np.random.randint(60, min(200, min(h, w)))
        
        if w - size <= 0 or h - size <= 0:
            break
        
        x = np.random.randint(0, w - size)
        y = np.random.randint(0, h - size)
        
        if not hay_solapamiento(x, y, size, size, bboxes, margen=25):
            region = gray[y:y+size, x:x+size]
            region_resized = cv2.resize(region, (128, 128))
            
            output_path = os.path.join(NEG_OUTPUT, f"neg_{contador:05d}.jpg")
            cv2.imwrite(output_path, region_resized)
            
            contador += 1
            
            if contador % 100 == 0:
                print(f"  → Generados: {contador}/{OBJETIVO_NEGATIVOS}")

print(f"\n✅ {contador} negativos generados en '{NEG_OUTPUT}/'")
print("\n" + "=" * 70)
print("SIGUIENTE PASO: py train.py")
print("=" * 70)