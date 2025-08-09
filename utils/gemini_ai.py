from PIL import Image
from google import genai
from google.genai import types

token_set = False
genai_token = None

def save_token(token):
    global token_set, genai_token
    genai_token = token
    token_set = True  
    masked_token = token[:4] + "*" * (len(token) - 4)
    if genai_token:
        return f"Your token: {masked_token}"
    else:
        return "Continue without token"

safety_set=[
    types.SafetySetting(
        category=types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
        threshold=types.HarmBlockThreshold.BLOCK_NONE,
    ),
    types.SafetySetting(
        category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
        threshold=types.HarmBlockThreshold.BLOCK_NONE,
    ),
    types.SafetySetting(
        category=types.HarmCategory.HARM_CATEGORY_HARASSMENT,
        threshold=types.HarmBlockThreshold.BLOCK_NONE,
    ),
    types.SafetySetting(
        category=types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
        threshold=types.HarmBlockThreshold.BLOCK_NONE,
    ),
]

def gemini_ai_ocr(imgs) :
    client = genai.Client(api_key=genai_token)
    image = Image.open(imgs)
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        config=types.GenerateContentConfig(
            system_instruction="You are an AI designed to extract text from provided images.  Maintain the original case of the extracted text. If the text in the image is uppercase, the extracted text should also be uppercase. If the text in the image is lowercase, the extracted text should be lowercase, and so on. Output only the extracted text without any additional words, explanations, or formatting changes.",
            safety_settings=safety_set,
        ),
            
        contents=["Please extract the text from each speech bubble. For each bubble, add a semicolon (;) at the end, and place the output on a new line for every speech bubble.", image])
    return response.text

def gemini_ai_translator(text) :
    client = genai.Client(api_key=genai_token)
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        config=types.GenerateContentConfig(
            system_instruction="You are an AI translator specializing in translating English to Indonesian. Your task is to translate the given text into clear and natural-sounding Indonesian while preserving the original capitalization. If the original text is uppercase, keep it uppercase; if lowercase, keep it lowercase. The translation should be easy to understand, not too formal, and should flow naturally in everyday conversation. Output only the translated text without any additional words or explanations.",
            safety_settings=safety_set,
            temperature = 0.5,
        ),
        contents=[f"Translate this text into Indonesian : {text}."])
    return response.text
