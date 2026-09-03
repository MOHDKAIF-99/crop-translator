import io
import os
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import google.generativeai as genai
from PIL import Image

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))


@app.post("/translate")
async def translate_image(
    image: UploadFile = File(...), target_lang: str = Form(...)
):
  contents = await image.read()
  img = Image.open(io.BytesIO(contents))

  model = genai.GenerativeModel("gemini-1.5-flash")

  prompt = f"""
    1. Extract all visible text from this cropped screenshot accurately.
    2. Translate and explain the extracted text into {target_lang}.
    
    Format output clearly as:
    **Original Text:**
    [Text]

    **Translation ({target_lang}):**
    [Translation]
    """

  response = model.generate_content([prompt, img])
  return {"result": response.text}