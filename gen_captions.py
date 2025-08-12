# gen_captions.py
import pathlib

IMG_DIR = pathlib.Path(r"H:\Python\LoRA_project\training_data\X_uniquename")
TRIGGER = "uniquename_person, portrait"

for img_path in IMG_DIR.glob("*.png"):
    txt_path = img_path.with_suffix(".txt")
    if txt_path.exists():
        continue  # don’t overwrite any you’ve already edited
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(TRIGGER + "\n")
print("Wrote default captions for", len(list(IMG_DIR.glob("*.txt"))), "images.")
