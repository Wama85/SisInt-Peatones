import cv2
import numpy as np
import os
import xml.etree.ElementTree as ET

# Carpetas de entrada
IMG_DIR = r"PEATONES/Train/JPEGImages"
XML_DIR = r"PEATONES/Train/Annotations"
NEG_OUTPUT = r"negativos"

# Crear carpeta de salida
os.makedirs(NEG_OUTPUT, exist_ok=True)

def get_bboxes(xml_path):
    """Extrae las bounding boxes de un archivo XML"""
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

def is_overlapping(x1, y1, x2, y2, boxes):
    """Verifica si una región se superpone con alguna bounding box"""
    for (bx1, by1, bx2, by2) in boxes:
        # Calcular intersección
        ix1 = max(x1, bx1)
        iy1 = max(y1, by1)
        ix2 = min(x2, bx2)
        iy2 = min(y2, by2)
        
        if ix1 < ix2 and iy1 < iy2:
            return True
    return False

# Procesar imágenes
contador = 0
max_negativos = 500  # Número máximo de negativos a generar

print("[INFO] Extrayendo regiones negativas...")

for img_name in os.listdir(IMG_DIR):
    if not img_name.lower().endswith(('.jpg', '.png', '.jpeg')):
        continue
    
    base_name = os.path.splitext(img_name)[0]
    img_path = os.path.join(IMG_DIR, img_name)
    xml_path = os.path.join(XML_DIR, base_name + '.xml')
    
    if not os.path.exists(xml_path):
        continue
    
    # Cargar imagen
    img = cv2.imread(img_path)
    if img is None:
        continue
    
    h, w = img.shape[:2]
    
    # Obtener bounding boxes de peatones
    boxes = get_bboxes(xml_path)
    
    # Extraer regiones aleatorias que NO contengan peatones
    intentos = 0
    max_intentos = 20
    
    while intentos < max_intentos and contador < max_negativos:
        intentos += 1
        
        # Tamaño aleatorio para la región negativa
        size = np.random.randint(64, 200)
        
        # Posición aleatoria
        x = np.random.randint(0, max(1, w - size))
        y = np.random.randint(0, max(1, h - size))
        
        # Verificar que NO se superponga con ningún peatón
        if not is_overlapping(x, y, x + size, y + size, boxes):
            # Extraer región
            region = img[y:y+size, x:x+size]
            
            # Redimensionar a 128x128
            region_resized = cv2.resize(region, (128, 128))
            
            # Guardar
            output_path = os.path.join(NEG_OUTPUT, f"neg_{contador:04d}.jpg")
            cv2.imwrite(output_path, region_resized)
            contador += 1
            
            if contador % 50 == 0:
                print(f"  Extraídos: {contador} negativos")
    
    if contador >= max_negativos:
        break

print(f"\n[OK] ✅ Extraídos {contador} ejemplos negativos en '{NEG_OUTPUT}/'")
print("\nAhora ejecuta: py train.py")