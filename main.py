import io
import os
import google.generativeai as genai
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

# Ensure GEMINI_API_KEY is set in your Render Environment Variables
genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/translate")
async def translate_image(
    image: UploadFile = File(...), target_lang: str = Form(...)
):
    try:
        contents = await image.read()
        img = Image.open(io.BytesIO(contents))

        # Use gemini-1.5-flash or gemini-2.0-flash
        model = genai.GenerativeModel("gemini-1.5-flash")

        prompt = f"""
        1. Extract all visible text from this cropped screenshot accurately.
        2. Translate the extracted text into {target_lang} (If target is Hinglish, write Hindi in Roman/English script).
        
        Format output clearly as:
        **Original Text:**
        [Extracted Text]
        
        **Translation ({target_lang}):**
        [Translation]
        """

        response = model.generate_content([prompt, img])
        return {"result": response.text}

    except Exception as e:
        return {"error": str(e)}