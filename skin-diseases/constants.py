import torch
from torchvision import transforms

NUM_EPOCHS = 3
BATCH_SIZE = 32
VALIDATION = False
DEVICE = ("cuda:0" if torch.cuda.is_available() else "cpu")
DATASET_PATH = f'/home/lilfil11/.cache/kagglehub/datasets/lilfil11/dermnet-dataset-for-year-project/versions/1'

PREPROCESS = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

DISEASES = {0: 'Light Diseases and Disorders of Pigmentation', 1: 'Lupus and other Connective Tissue diseases',
            2: 'Acne and Rosacea Photos', 3: 'Systemic Disease', 4: 'Poison Ivy Photos and other Contact Dermatitis',
            5: 'Vascular Tumors', 6: 'Urticaria Hives', 7: 'Atopic Dermatitis Photos', 8: 'Bullous Disease Photos',
            9: 'Hair Loss Photos Alopecia and other Hair Diseases', 10: 'Tinea Ringworm Candidiasis and other Fungal Infections',
            11: 'Psoriasis pictures Lichen Planus and related diseases', 12: 'Melanoma Skin Cancer Nevi and Moles',
            13: 'Nail Fungus and other Nail Disease', 14: 'Scabies Lyme Disease and other Infestations and Bites',
            15: 'Eczema Photos', 16: 'Exanthems and Drug Eruptions', 17: 'Herpes HPV and other STDs Photos',
            18: 'Seborrheic Keratoses and other Benign Tumors', 19: 'Actinic Keratosis Basal Cell Carcinoma and other Malignant Lesions',
            20: 'Vasculitis Photos', 21: 'Cellulitis Impetigo and other Bacterial Infections', 22: 'Warts Molluscum and other Viral Infections'}
