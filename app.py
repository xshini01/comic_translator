from add_text import add_text
from detect_bubbles import detect_bubbles
from process_bubble import process_bubble
from qwen2_vl_ocr import qwen2_vl_ocr
import gemini_ai
from translator import MangaTranslator
from IPython.display import clear_output
from ultralytics import YOLO
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from PIL import Image
import gradio as gr
import numpy as np
from google.genai.errors import ClientError
import cv2
import time

TITLE = "Komik Translator"
DESCRIPTION = "Translate komik dari Inggris => Indonesia"


with gr.Blocks() as token_interface:
        gr.Markdown("## Token Gemini Ai (opsional)")
        token_input = gr.Textbox(
            label="Jika Anda menggunakan token Gemini AI, OCR dan terjemahan akan dilakukan menggunakan Gemini AI. Jika tidak, model default akan digunakan.",
            info= "Anda bisa mendapatkan token Gemini AI dari aistudio.google.com/apikey. Token ini bersifat opsional dan dapat digunakan untuk pemindaian dan terjemahan teks menggunakan Gemini AI (Google).",
            placeholder="Masukan token disini (opsional) ...",
            type="password"
        )
        save_button = gr.Button("Submit", variant="primary")
        output_label = gr.Label(label= "your token :")
        save_button.click(fn=gemini_ai.save_token, inputs=token_input, outputs=output_label)
    
token_interface.launch()

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


def predict(img, MODEL, translation_method, font, progress=gr.Progress(track_tqdm=True)):
    if translation_method == None:
        translation_method = "google"
    if font == None:
        font = "fonts/fonts_animeace_i.ttf"

    results = detect_bubbles(MODEL, img)

    manga_translator = MangaTranslator()

    image = np.array(img)

    for result in results:
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

    return image

demo = gr.Interface(fn=predict,
                    inputs=["image",
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
                    outputs=[gr.Image()],
                    title=TITLE,
                    description=DESCRIPTION)


demo.launch(debug=True, share=True, inline=False)
