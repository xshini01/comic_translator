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
        model="gemini-2.5-flash-lite",
        config=types.GenerateContentConfig(
            system_instruction="Extract text from images exactly as-is, preserving original case. Output only the extracted text, nothing else.",
            safety_settings=safety_set,
        ),
            
        contents=["Extract each speech bubble's text. End each with ';', one per line.", image])
    return response.text

def gemini_ai_translator(text):
    client = genai.Client(api_key=genai_token)
    response = client.models.generate_content(
        model="gemini-2.5-flash-lite",
        config=types.GenerateContentConfig(
            system_instruction="Translate English to Indonesian. Keep original capitalization (uppercase stays uppercase, lowercase stays lowercase). Output must be natural, casual, and easy to understand. Output only the translation, nothing else.",
            safety_settings=safety_set,
            temperature=0.5,
        ),
        contents=[f"Translate to Indonesian: {text}"])
    return response.text
