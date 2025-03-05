import os
from PIL import Image
from reportlab.pdfgen import canvas
import gradio as gr

def compress_toPDF():

    folder_path = "save_images"
    output_folder = "Hasil-PDF"
    os.makedirs(output_folder, exist_ok=True)
    pdf_file = os.path.join(output_folder, f"compressed_images_{sum(1 for entry in os.scandir(output_folder) if entry.is_file()) + 1}.pdf")

    images = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(("png", "jpg", "jpeg"))]
    
    if images:
        img = Image.open(images[0])
        img.convert("RGB").save(pdf_file, save_all=True, append_images=[Image.open(i).convert("RGB") for i in images[1:]])

        print(f"PDF file '{pdf_file}' berhasil dibuat!")
    else:
        print("Tidak ada gambar yang ditemukan!")
    
    return pdf_file