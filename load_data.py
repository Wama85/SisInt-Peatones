import cv2
import os
import xml.etree.ElementTree as ET
import numpy as np

# nuevo extractor 7x7
def extract_7x7(face):
    small = cv2.resize(face, (7,7))
    small = small.astype("float32") / 255.0
    return small.flatten()

def load_face_from_xml(img_path, xml_path):
    img = cv2.imread(img_path)
    if img is None:
        print("[ERROR] Imagen no encontrada:", img_path)
        return None

    h_img, w_img = img.shape[:2]

    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
    except:
        print("[ERROR] XML no válido:", xml_path)
        return None

    w_xml = int(root.find("size/width").text)
    h_xml = int(root.find("size/height").text)

    scale_x = w_img / w_xml
    scale_y = h_img / h_xml

    for obj in root.findall("object"):
        bb = obj.find("bndbox")

        xmin = int(bb.find("xmin").text) * scale_x
        ymin = int(bb.find("ymin").text) * scale_y
        xmax = int(bb.find("xmax").text) * scale_x
        ymax = int(bb.find("ymax").text) * scale_y

        xmin = int(max(0, xmin))
        ymin = int(max(0, ymin))
        xmax = int(min(w_img-1, xmax))
        ymax = int(min(h_img-1, ymax))

        if xmax <= xmin or ymax <= ymin:
            return None

        face = img[ymin:ymax, xmin:xmax]

        if face.size == 0:
            return None

        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        return face

    return None

def load_dataset(img_folder, xml_folder, txt_file):
    X, Y = [], []

    with open(txt_file, "r", encoding="utf-8") as f:
        names = f.read().splitlines()

    for name in names:
        img_path = os.path.join(img_folder, name + ".jpg")
        xml_path = os.path.join(xml_folder, name + ".xml")

        face = load_face_from_xml(img_path, xml_path)
        if face is None:
            continue

        desc = extract_7x7(face)
        X.append(desc)
        Y.append(1)

    X = np.array(X).T
    Y = np.array(Y).reshape(1, -1)

    print("[INFO] Dataset cargado:", X.shape[1], "rostros")
    return X, Y
