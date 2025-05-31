from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import onnxruntime as ort
import numpy as np
from PIL import Image
import io

app = FastAPI()

model_path = "model/blur_classifier.onnx"
session = ort.InferenceSession(model_path)

def preprocess_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize((224, 224))
    img_data = np.array(image).astype(np.float32) / 255.0

    # Transpõe para (C, H, W)
    img_data = np.transpose(img_data, (2, 0, 1))

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3,1,1)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3,1,1)
    img_data = (img_data - mean) / std

    img_data = np.expand_dims(img_data, axis=0)
    return img_data

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Arquivo não é uma imagem")
    image_bytes = await file.read()
    input_tensor = preprocess_image(image_bytes)
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: input_tensor})

    probs = outputs[0][0]
    print("Probabilidades:", probs)  # Para debug no console do Render

    predicted_class = int(np.argmax(probs))
    confidence = float(np.max(probs))

    return JSONResponse(content={
        "predicted_class": predicted_class,
        "confidence": confidence
    })

@app.get("/")
def root():
    return {"status": "API ONNX funcionando"}


