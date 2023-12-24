from fastapi import FastAPI, UploadFile, File
import uvicorn
from PIL import Image
from io import BytesIO
import numpy as np
import tensorflow as tf

app = FastAPI()
MODEL = tf.keras.models.load_model("../../saved_models/1")
CLASS_NAMES = ["Tomato Early Blight", "Tomato Late Blight", "Tomato Healthy"]


def convert_data_to_image(data):
    image = np.array(Image.open(BytesIO(data)))
    return image


def predict_img(image):
    pred = MODEL.predict(image)[0]
    class_name = CLASS_NAMES[np.argmax(pred)]
    confidence = np.round(np.max(pred) * 100, 2)
    return class_name, confidence


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image = convert_data_to_image(await file.read())
    image = np.expand_dims(image, axis=0)
    class_name, confidence = predict_img(image)
    return {"class": class_name, "confidence": confidence}


if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)
