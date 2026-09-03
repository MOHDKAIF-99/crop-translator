import io
import os
import google.generativeai as genai
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from PIL import Image

genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def read_root():
    return FileResponse("index.html")

@app.post("/translate")
async def translate_image(
    image: UploadFile = File(...), target_lang: str = Form(...)
):
    try:
        contents = await image.read()
        img = Image.open(io.BytesIO(contents))

        prompt = f"""
        1. Extract all visible text from this cropped screenshot accurately.
        2. Translate the extracted text into {target_lang} (If target is Hinglish, write Hindi in Roman/English script).
        
        Format output clearly as:
        **Original Text:**
        [Extracted Text]
        
        **Translation ({target_lang}):**
        [Translation]
        """

        # Auto-tries active Gemini models to prevent 404 errors
        model_names = ["gemini-1.5-flash-latest" ,"gemini-3.6-flash "]
        
        last_error = None
        for name in model_names:
            try:
                model = genai.GenerativeModel(name)
                response = model.generate_content([prompt, img])
                return {"result": response.text}
            except Exception as e:
                last_error = e
                continue

        return {"error": f"Model error: {str(last_error)}"}

    except Exception as e:
        return {"error": str(e)}