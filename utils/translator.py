from deep_translator import GoogleTranslator, DeeplTranslator
from transformers import pipeline
# import translators as ts


class MangaTranslator:
    def __init__(self):
        self.target = "id"
        self.source = "en"
        self.translators = {
            "google": self._translate_with_google,
            "hf": self._translate_with_hf,
            # "baidu": self._translate_with_baidu,
            # "bing": self._translate_with_bing,
            "deepl": self._translate_with_deepl
        }

    def translate(self, text, method="google", api=None):
        """
        Translates the given text to the target language using the specified method.

        Args:
            text (str): The text to be translated.
            method (str): "google", "hf", or "deepl"
            api (str, optional): API key required for Deepl, if used.

        Returns:
            str: The translated text.
        """
        translator_func = self.translators.get(method)

        if translator_func:
            return translator_func(self._preprocess_text(text), api)
        else:
            raise ValueError("Invalid translation method.")

    def _translate_with_google(self, text, api=None):
        translator = GoogleTranslator(source=self.source, target=self.target)
        translated_text = translator.translate(text)
        return translated_text if translated_text is not None else text

    def _translate_with_hf(self, text, api=None):
        pipe = pipeline("translation", model="Helsinki-NLP/opus-mt-en-id")
        translated_text = pipe(text)[0]["translation_text"]
        return translated_text if translated_text is not None else text

    # def _translate_with_baidu(self, text, api=None):
    #     translated_text = ts.translate_text(text, translator="baidu",
    #                                         from_language=self.source, 
    #                                         to_language=self.target)
    #     return translated_text if translated_text is not None else text

    # def _translate_with_bing(self, text, api=None):
    #     translated_text = ts.translate_text(text, translator="bing",
    #                                         from_language=self.source, 
    #                                         to_language=self.target)
    #     return translated_text if translated_text is not None else text

    def _translate_with_deepl(self, text, api=None):
        if not api:
            raise ValueError("DeepL API key must be provided.")
        translated_text = DeeplTranslator(api_key=api, source="en", target="id", use_free_api=True).translate(text)
        return translated_text if translated_text is not None else text

    def _preprocess_text(self, text):
        return text.replace("．", ".")
