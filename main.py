import io
import os
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import google.generativeai as genai

app = FastAPI()

# Enable CORS so browser requests from local files are allowed
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))

@app.get("/")
def home():
    return {"status": "Crop Translator API is running"}

@app.post("/translate")
async def translate_image(
    image: UploadFile = File(...), target_lang: str = Form(...)
):
    contents = await image.read()
    img = Image.open(io.BytesIO(contents))

    model = genai.GenerativeModel("gemini-2.5-flash")

    prompt = f"""
    1. Extract all visible text from this cropped screenshot accurately.
    2. Translate the extracted text into {target_lang}.

    Format output clearly as:
    **Original Text:**
    [Text]

    **Translation ({target_lang}):**
    [Translation]
    """

    response = model.generate_content([prompt, img])
    return {"result": response.text}