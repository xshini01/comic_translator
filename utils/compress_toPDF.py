import os
import img2pdf
from natsort import natsorted

def compress_toPDF():
    folder_path = "save_images"
    output_folder = "Hasil-PDF"
    os.makedirs(output_folder, exist_ok=True)

    pdf_file = os.path.join(
        output_folder,
        f"compressed_images_{sum(1 for entry in os.scandir(output_folder) if entry.is_file()) + 1}.pdf"
    )

    images = natsorted([
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if f.lower().endswith(("png", "jpg", "jpeg"))
    ])

    if images:
        with open(pdf_file, "wb") as f:
            f.write(img2pdf.convert(images))
        print(f"PDF file '{pdf_file}' berhasil dibuat!")
    else:
        print("Tidak ada gambar yang ditemukan!")

    return pdf_file
