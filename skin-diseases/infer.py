from fastapi import FastAPI, UploadFile
from torch import device, load
from torch.nn import Linear
from torchvision.io import read_image
from torchvision.models import efficientnet_v2_s
from train import PREPROCESS

DISEASES = {
    0: "Light Diseases and Disorders of Pigmentation",
    1: "Lupus and other Connective Tissue diseases",
    2: "Acne and Rosacea Photos",
    3: "Systemic Disease",
    4: "Poison Ivy Photos and other Contact Dermatitis",
    5: "Vascular Tumors",
    6: "Urticaria Hives",
    7: "Atopic Dermatitis Photos",
    8: "Bullous Disease Photos",
    9: "Hair Loss Photos Alopecia and other Hair Diseases",
    10: "Tinea Ringworm Candidiasis and other Fungal Infections",
    11: "Psoriasis pictures Lichen Planus and related diseases",
    12: "Melanoma Skin Cancer Nevi and Moles",
    13: "Nail Fungus and other Nail Disease",
    14: "Scabies Lyme Disease and other Infestations and Bites",
    15: "Eczema Photos",
    16: "Exanthems and Drug Eruptions",
    17: "Herpes HPV and other STDs Photos",
    18: "Seborrheic Keratoses and other Benign Tumors",
    19: "Actinic Keratosis Basal Cell Carcinoma and other Malignant Lesions",
    20: "Vasculitis Photos",
    21: "Cellulitis Impetigo and other Bacterial Infections",
    22: "Warts Molluscum and other Viral Infections",
}


app = FastAPI(title="Skin diseases classification")

# Модель
model = efficientnet_v2_s()
model.classifier[1] = Linear(1280, 23)
model.load_state_dict(
    load("../models/EfficientNetV2-S.pth", map_location=device("cpu"))
)
model.eval()


# Корневая директория
@app.get("/")
def root():
    return {
        "status": "successful",
        "message": "Hello! This is a web-site for classifying images with "
        "skin diseases using deep learning models.",
    }


# Получение предсказания для загруженного изображения
@app.post("/predict")
def predict(file_uploaded: UploadFile) -> dict:
    file_location = f"tmp_files/{file_uploaded.filename}"
    with open(file_location, "wb+") as file_object:
        file_object.write(file_uploaded.file.read())

    image = read_image(file_location) / 255
    image = PREPROCESS(image).unsqueeze(0)
    prediction = model(image)

    return {
        "status": "successful",
        "filename": file_uploaded.filename,
        "predict": f"Your disease is {DISEASES[prediction.argmax(1).item()]}",
    }
