import uvicorn
import pickle
import pytesseract
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io

app = FastAPI()

# Enable connection from React
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- LOAD THE TRAINED MODEL ---
print("Loading AI Models...")
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)
with open('remedies.pkl', 'rb') as f:
    remedies = pickle.load(f)
print("AI Models Loaded Successfully.")

# --- MODULE A: SYMPTOM CHECKER ---
@app.post("/predict")
async def predict(symptoms: str = Form(...)):
    # 1. Convert text to numbers using the saved vectorizer
    vectorized_input = vectorizer.transform([symptoms])
    
    # 2. Predict
    prediction = model.predict(vectorized_input)[0]
    
    # 3. Get Confidence Score
    probs = model.predict_proba(vectorized_input)[0]
    confidence = max(probs) * 100
    
    # 4. Get Remedy
    advice = remedies.get(prediction, "Consult a doctor.")

    return {
        "disease": prediction,
        "confidence": f"{confidence:.2f}%",
        "remedy": advice
    }

# --- MODULE B: OCR (PRESCRIPTIONS) ---
# Uncomment line below if on Windows
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

@app.post("/scan_report")
async def scan_report(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))
    
    # Extract text
    text = pytesseract.image_to_string(image)
    
    return {"extracted_text": text}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)