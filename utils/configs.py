import os
from utils import gemini_ai

class Translator:
    def __init__(self):
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))

        self.models = {
            "Model-1": os.path.join(BASE_DIR, "..", "model.pt"),
            "Model-2": os.path.join(BASE_DIR, "..", "best.pt")
        }

        self.full_methods = {
            "Google": "google",
            "Helsinki-NLP's opus-mt-en-id model": "hf",
            "Gemini AI": "gemini",
            "DeepL": "deepl",
        }

        self.fonts = {
            "animeace_i": os.path.join(BASE_DIR, "..", "fonts", "fonts_animeace_i.ttf"),
            "mangati": os.path.join(BASE_DIR, "..", "fonts", "fonts_mangati.ttf"),
            "ariali": os.path.join(BASE_DIR, "..", "fonts", "fonts_ariali.ttf"),
        }

    def get_available_methods(self):
        methods = self.full_methods.copy()
        if not gemini_ai.genai_token:
            methods.pop("Gemini AI", None)
        return list(methods.keys())