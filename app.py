from utils.add_text import add_text
from utils.detect_bubbles import detect_bubbles
from utils.process_bubble import process_bubble
from utils.qwen2_vl_ocr import qwen2_vl_ocr
from utils.extract_file import extract_file
from utils.compress_toPDF import compress_toPDF
from utils import gemini_ai
from utils.translator import MangaTranslator
from IPython.display import clear_output
from ultralytics import YOLO
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from PIL import Image
import gradio as gr
import numpy as np
from google.genai.errors import ClientError
import cv2
import time
import os
from tqdm import tqdm
import shutil
from natsort import natsorted

def get_images(image_folder):
    image_paths = [
        os.path.join(image_folder, file)
        for file in os.listdir(image_folder)
        if file.lower().endswith((".png", ".jpg", ".jpeg"))
    ]
    return image_paths  # Mengembalikan daftar path gambar

# fungsi mencoba api
def retry_on_429(func, *args, max_retries=5, base_wait=5, **kwargs):
    """Retry jika terjadi error 429 (RESOURCE_EXHAUSTED) dengan exponential backoff."""
    retries = 0

    while retries < max_retries:
        try:
            return func(*args, **kwargs)
        except ClientError as e:
            error_message = str(e)  
            if 'RESOURCE_EXHAUSTED' in error_message or '429' in error_message:
                retries += 1
                wait_time = base_wait * (2 ** (retries - 1)) 
                print(f"[ERROR 429] Token habis. Coba lagi dalam {wait_time} detik... ({retries}/{max_retries})")
                time.sleep(wait_time)
            else:
                raise  
        except Exception as e:
            print(f"Error lain: {e}")
            break

    raise RuntimeError(f"Gagal setelah {max_retries} percobaan karena kehabisan token.")


# main fungsi
def predict(files_input, MODEL, translation_method, font, progress=gr.Progress(track_tqdm=True)):
    source_dir = 'folder_ekstrak'
    save_dir = "save_images"

    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
        shutil.rmtree(source_dir)

    os.makedirs(save_dir, exist_ok=True)
    
    if translation_method is None:
        translation_method = "google"
    if font is None:
        font = "fonts/fonts_animeace_i.ttf"
    
    extract_file(files_input)

    manga_translator = MangaTranslator()

    for root, dirs, files in os.walk(source_dir):
        files = natsorted(files)
        for file in tqdm(files, desc="Memproses Gambar"):
            file_path = os.path.join(root, file)

            results = detect_bubbles(MODEL, file_path)

            image = np.array(Image.open(file_path).convert("RGB"))

            for result in tqdm(results, desc= "Mentranslate Gambar"):
                x1, y1, x2, y2, score, class_id = result

                detected_image = image[int(y1):int(y2), int(x1):int(x2)]

                im = Image.fromarray(np.uint8((detected_image)*255))

                im.save("detected_image.png")

                detected_image, cont = process_bubble(detected_image)

                if gemini_ai.genai_token :
                    text = retry_on_429(gemini_ai.gemini_ai_ocr, "detected_image.png")
                    text_translated = retry_on_429(gemini_ai.gemini_ai_translator, text)
                else:
                    text = qwen2_vl_ocr(im, model_ocr, processor_ocr)
                    text_translated = manga_translator.translate(text,
                                                                method=translation_method)

                image_with_text = add_text(detected_image, text_translated, font, cont)

            image_path = os.path.join(save_dir, f"output_image_{sum(1 for entry in os.scandir(save_dir) if entry.is_file()) + 1}.png")
            cv2.imwrite(image_path, image)
            time.sleep(0.1)

    to_pdf= compress_toPDF()

    return get_images(source_dir), get_images(save_dir), gr.update(value=to_pdf, visible=True)

            
TITLE = "Komik Translator"
DESCRIPTION = "Translate komik dari Inggris => Indonesia"

# ui token
with gr.Blocks() as token_interface:
        gr.Markdown("## Token/API Key Gemini Ai (opsional)")
        token_input = gr.Textbox(
            label="Jika Anda menggunakan Token/API Key Gemini AI, OCR dan terjemahan akan dilakukan menggunakan Gemini AI. Jika tidak, model default(qwen2_vl_ocr) akan digunakan.",
            info="Anda bisa mendapatkan Token/API Key Gemini AI dari <a href=\"https://aistudio.google.com/apikey\" target=\"_blank\"> SINI (Token/API Key Google)</a>. \nToken/API Key ini bersifat opsional dan dapat digunakan untuk pemindaian dan terjemahan teks menggunakan Gemini AI (Google).",
            placeholder="Masukan Token/API Key disini (opsional) ...",
            type="password"
        )
        save_button = gr.Button("Submit", variant="primary")
        output_label = gr.Label(label= "your Token/API Key :")
        save_button.click(fn=gemini_ai.save_token, inputs=token_input, outputs=output_label)

clear_output()
token_interface.launch(share=True)

while not gemini_ai.token_set:
    time.sleep(2)

model_ocr, processor_ocr = None, None

if not gemini_ai.genai_token:
    def load_ocr_model():
        global model_ocr, processor_ocr
        if model_ocr is None or processor_ocr is None:
            model_ocr = Qwen2VLForConditionalGeneration.from_pretrained(
                "prithivMLmods/Qwen2-VL-OCR-2B-Instruct", torch_dtype="auto", device_map="auto"
            )
            processor_ocr = AutoProcessor.from_pretrained("prithivMLmods/Qwen2-VL-OCR-2B-Instruct")
    
    load_ocr_model()

# main interface
demo = gr.Interface(fn=predict,
                    inputs=[gr.Files(),
                            gr.Dropdown([("model-1", "model.pt"),
                                         ("model-2","best.pt")],
                                        label="Model YOLO",
                                        value="best.pt"),
                            gr.Dropdown([("Google", "google"),
                                         ("DeepL", "deepl"),
                                         ("Helsinki-NLP's opus-mt-en-id model",
                                          "hf"),
                                         ("Baidu", "baidu"),
                                         ("Bing", "bing")],
                                        label="Translation Method",
                                        value="google"),
                            gr.Dropdown([("animeace_i", "fonts/fonts_animeace_i.ttf"),
                                         ("mangati", "fonts/fonts_mangati.ttf"),
                                         ("ariali", "fonts/fonts_ariali.ttf")],
                                        label="Text Font",
                                        value="fonts/fonts_animeace_i.ttf")
                            ],
                    outputs=[gr.Gallery(label="Gambar Asli"),
                             gr.Gallery(label=" Hasil Terjemahan"),
                             gr.File(label="Download File", visible= False)],
                    title=TITLE,
                    description=DESCRIPTION)

clear_output()
demo.launch(debug=True, share=True, inline=False)
