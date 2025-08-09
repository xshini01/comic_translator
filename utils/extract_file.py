import os
import zipfile
import rarfile
import shutil
from pdf2image import convert_from_path
from PIL import Image

def extract_file(files):
    extract_to = 'folder_ekstrak'
    os.makedirs(extract_to, exist_ok=True)

    for file in files:
        if  os.path.isdir(file):
            pass
        elif file.endswith(".zip"):
            try:
                with zipfile.ZipFile(file, 'r') as zip_ref:
                    zip_ref.extractall(extract_to)
                print(f"Berhasil mengekstrak {file}")
            except zipfile.BadZipFile:
                print(f"File ZIP {file} rusak atau tidak valid.")

        elif file.endswith(".rar"):
            try:
                with rarfile.RarFile(file, 'r') as rar_ref:
                    rar_ref.extractall(extract_to)
                print(f"Berhasil mengekstrak {file}")
            except rarfile.Error as e:
                print(f"Gagal mengekstrak {file}: {e}")

        elif file.endswith(".pdf"):
            try :
                images = convert_from_path(file)

                for i, image in enumerate(images):
                    image_path = os.path.join(extract_to, f"page_{i+1}.png")  # Bisa diganti ke .png
                    image.save(image_path, "PNG")
                print(f"Berhasil mengekstrak {file}")
            except Exception as e:
                print(f"Terjadi kesalahan: {e}")

        else:
            try:
                with Image.open(file) as img:
                    img.verify()  # Cek apakah file valid sebagai gambar
                shutil.move(file, os.path.join(extract_to, os.path.basename(file)))
            except Exception:
                print(f"File {file} bukan gambar atau tidak valid.")

