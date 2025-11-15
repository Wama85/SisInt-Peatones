import os

ann_folder = "PEATONES/Train/Annotations"
out_txt = "PEATONES/Train/train.txt"

xml_files = sorted(os.listdir(ann_folder))

with open(out_txt, "w", encoding="utf-8") as f:
    for file in xml_files:
        if file.endswith(".xml"):
            name = file[:-4]  # quitar ".xml"
            f.write(name + "\n")

print("✔ train.txt generado correctamente")
