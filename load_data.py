import cv2
import numpy as np
import xml.etree.ElementTree as ET
import os

# ========== FUNCIONES DE CONVOLUCIÓN ==========

def aplicar_convolucion(imagen, kernel):
    img_h, img_w = imagen.shape
    ker_h, ker_w = kernel.shape
    
    out_h = img_h - ker_h + 1
    out_w = img_w - ker_w + 1
    output = np.zeros((out_h, out_w))
    
    for i in range(out_h):
        for j in range(out_w):
            region = imagen[i:i+ker_h, j:j+ker_w]
            output[i, j] = np.sum(region * kernel)
    
    return output

def max_pooling(imagen, pool_size=2):
    img_h, img_w = imagen.shape
    out_h = img_h // pool_size
    out_w = img_w // pool_size
    output = np.zeros((out_h, out_w))
    
    for i in range(out_h):
        for j in range(out_w):
            start_i = i * pool_size
            start_j = j * pool_size
            region = imagen[start_i:start_i+pool_size, start_j:start_j+pool_size]
            output[i, j] = np.max(region)
    
    return output

# ========== EXTRACTORES DE CARACTERÍSTICAS ==========

def extract_conv_features(img):
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    img = cv2.resize(img, (64, 64))
    img = img.astype(np.float32) / 255.0
    
    kernel_sobel_x = np.array([
        [-1, 0, +1],
        [-2, 0, +2],
        [-1, 0, +1]
    ], dtype=np.float32)
    
    kernel_sobel_y = np.array([
        [-1, -2, -1],
        [ 0,  0,  0],
        [+1, +2, +1]
    ], dtype=np.float32)
    
    kernel_laplacian = np.array([
        [ 0, -1,  0],
        [-1,  4, -1],
        [ 0, -1,  0]
    ], dtype=np.float32)
    
    conv_x = aplicar_convolucion(img, kernel_sobel_x)
    conv_x = np.maximum(0, conv_x)
    pool_x = max_pooling(conv_x, pool_size=2)
    
    conv_y = aplicar_convolucion(img, kernel_sobel_y)
    conv_y = np.maximum(0, conv_y)
    pool_y = max_pooling(conv_y, pool_size=2)
    
    conv_lap = aplicar_convolucion(img, kernel_laplacian)
    conv_lap = np.maximum(0, conv_lap)
    pool_lap = max_pooling(conv_lap, pool_size=2)
    
    features_x = pool_x.flatten()
    features_y = pool_y.flatten()
    features_lap = pool_lap.flatten()
    
    features = np.concatenate([features_x, features_y, features_lap])
    
    return features

def extract_7x7(img):
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    img_resized = cv2.resize(img, (7, 7))
    return img_resized.flatten()

# ========== CARGA DE DATASET ==========

def load_dataset(img_dir, xml_dir, txt_path, use_conv=False):
    X, Y = [], []
    
    print(f"[INFO] Método: {'Convolución (Sobel+Laplaciano)' if use_conv else '7x7 simple'}")
    
    with open(txt_path, 'r', encoding='utf-8') as f:
        for line in f:
            img_name = line.strip()
            if not img_name:
                continue
            
            img_path = os.path.join(img_dir, img_name + '.jpg')
            xml_path = os.path.join(xml_dir, img_name + '.xml')
            
            if not os.path.exists(img_path) or not os.path.exists(xml_path):
                continue
            
            img = cv2.imread(img_path)
            if img is None:
                continue
            
            tree = ET.parse(xml_path)
            root = tree.getroot()
            
            for obj in root.findall('object'):
                bbox = obj.find('bndbox')
                xmin = int(bbox.find('xmin').text)
                ymin = int(bbox.find('ymin').text)
                xmax = int(bbox.find('xmax').text)
                ymax = int(bbox.find('ymax').text)
                
                roi = img[ymin:ymax, xmin:xmax]
                
                if roi.size == 0:
                    continue
                
                try:
                    if use_conv:
                        features = extract_conv_features(roi)
                    else:
                        features = extract_7x7(roi)
                    
                    X.append(features)
                    Y.append(1)
                
                except Exception as e:
                    continue
    
    if len(X) == 0:
        return np.empty((0, 0)), np.empty((1, 0))
    
    X = np.array(X).T
    Y = np.array(Y).reshape(1, -1)
    
    print(f"[INFO] Dataset cargado: {X.shape[1]} peatones")
    print(f"[INFO] Características por imagen: {X.shape[0]}")
    
    return X, Y