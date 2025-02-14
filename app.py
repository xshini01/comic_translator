from add_text import add_text
from detect_bubbles import detect_bubbles
from process_bubble import process_bubble
from qwen2_vl_ocr import qwen2_vl_ocr
from translator import MangaTranslator
from IPython.display import clear_output
from ultralytics import YOLO
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from PIL import Image
import gradio as gr
import numpy as np
import cv2

# Load OCR Model
model_ocr = Qwen2VLForConditionalGeneration.from_pretrained(
    "prithivMLmods/Qwen2-VL-OCR-2B-Instruct", torch_dtype="auto", device_map="auto"
)
processor_ocr = AutoProcessor.from_pretrained("prithivMLmods/Qwen2-VL-OCR-2B-Instruct")

clear_output()
print("Setup Complete")

TITLE = "Komik Translator"
DESCRIPTION = "Translate komik dari Inggris => Indonesia"


def predict(img, MODEL, translation_method, font):
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

        text = qwen2_vl_ocr(detected_image, model_ocr, processor_ocr)

        detected_image, cont = process_bubble(detected_image)

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
