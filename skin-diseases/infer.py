from torch import load, device
from torch.nn import Linear
from torchvision.io import read_image
from torchvision.models import efficientnet_v2_s

from fastapi import FastAPI, Path, UploadFile

from constants import *


app = FastAPI(title='Skin diseases classification')


# Модель
model = efficientnet_v2_s()
model.classifier[1] = Linear(1280, 23)
model.load_state_dict(load('../models/EfficientNetV2-S.pth', map_location=device('cpu')))
model.eval()


# Корневая директория
@app.get('/')
def root():
    return {'status': 'successful',
            'message': 'Hello! This is a web-site for classifying images with '
                       'skin diseases using deep learning models.'}


# Получение предсказания для загруженного изображения
@app.post('/predict')
def predict(file_uploaded: UploadFile) -> dict:
    file_location = f"tmp_files/{file_uploaded.filename}"
    with open(file_location, "wb+") as file_object:
        file_object.write(file_uploaded.file.read())

    image = read_image(file_location) / 255
    image = PREPROCESS(image).unsqueeze(0)
    prediction = model(image)

    global images_loaded
    images_loaded += 1

    return {'status': 'successful',
            'filename': file_uploaded.filename,
            'predict': f'Your disease is {DISEASES[prediction.argmax(1).item()]}'}
