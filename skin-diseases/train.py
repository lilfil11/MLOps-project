import hydra
import numpy as np
import mlflow
import torch
from dataset import MedicalImageDataset
from metrics import test_metrics
from mlflow.models import infer_signature
from omegaconf import DictConfig
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.models import EfficientNet_V2_S_Weights, efficientnet_v2_s
from tqdm import tqdm

PREPROCESS = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


def train(dataloader, model, criterion, optimizer, scheduler):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)

    model.train()
    loss_avg, accuracy_avg = 0, 0
    for data, labels in tqdm(dataloader):
        data, labels = data.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        predictions = model(data)
        loss = criterion(predictions, labels.long())
        accuracy = (predictions.argmax(1) == labels).type(torch.float).sum().item()
        loss.backward()
        optimizer.step()

        loss_avg += loss.item()
        accuracy_avg += accuracy

    if scheduler is not None:
        scheduler.step()

    return loss_avg / num_batches, accuracy_avg / size


def test(dataloader, model, criterion):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)

    model.eval()
    with torch.no_grad():
        loss_avg, accuracy_avg = 0, 0

        for data, labels in dataloader:
            data, labels = data.to(DEVICE), labels.to(DEVICE)

            predictions = model(data)
            loss_avg += criterion(predictions, labels.long()).item()
            accuracy_avg += (
                (predictions.argmax(1) == labels).type(torch.float).sum().item()
            )

    return loss_avg / num_batches, accuracy_avg / size


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(config: DictConfig):
    # Датасет
    train_dataset = MedicalImageDataset(
        img_dir=config["dataset"]["dataset_path"], data="train", transform=PREPROCESS
    )
    test_dataset = MedicalImageDataset(
        img_dir=config["dataset"]["dataset_path"], data="test", transform=PREPROCESS
    )
    if config["training"]["validation"]:
        train_idx, val_idx = train_test_split(
            np.arange(len(train_dataset)), test_size=0.2, shuffle=True, random_state=0
        )
        train_dataset = Subset(train_dataset, train_idx)
        val_dataset = Subset(train_dataset, val_idx)
        val_dataloader = DataLoader(
            val_dataset, batch_size=config["training"]["batch_size"], shuffle=False
        )

    train_dataloader = DataLoader(
        train_dataset, batch_size=config["training"]["batch_size"], shuffle=True
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size=config["training"]["batch_size"], shuffle=False
    )

    # Создаём эксперимент в MLFlow для логгирования
    mlflow.set_tracking_uri(uri=config["mlflow"]["tracking_uri"])

    exp_name = 'kirill_filatov'
    try:
        exp_id = mlflow.create_experiment(name=exp_name)
    except:
        mlflow.set_experiment(exp_name)
        exp_id = dict(mlflow.get_experiment_by_name(exp_name))['experiment_id']

    # Обучаем модель
    with mlflow.start_run(run_name='Lilfil11', experiment_id=exp_id) as parent_run:
        weights = EfficientNet_V2_S_Weights.DEFAULT
        model = efficientnet_v2_s(weights=weights).to(DEVICE)
        model.classifier[1] = nn.Linear(1280, 23).to(DEVICE)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=3e-4, weight_decay=5e-5)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

        train_loss, train_accuracy = (
            np.zeros(config["training"]["num_epochs"]),
            np.zeros(config["training"]["num_epochs"]),
        )
        val_loss, val_accuracy = (
            np.zeros(config["training"]["num_epochs"]),
            np.zeros(config["training"]["num_epochs"]),
        )

        for epoch in range(config["training"]["num_epochs"]):
            train_loss[epoch], train_accuracy[epoch] = train(
                train_dataloader, model, criterion, optimizer, scheduler
            )
            print(
                f"\n>>> Epoch {epoch + 1}\n\tTrain\tloss: {train_loss[epoch]:.4f},  accuracy: {100 * train_accuracy[epoch]:.2f}"
            )

            if config["training"]["validation"]:
                val_loss[epoch], val_accuracy[epoch] = test(
                    val_dataloader, model, criterion
                )
                print(
                    f"\tVal  \tloss: {val_loss[epoch]:.4f},  accuracy: {100 * val_accuracy[epoch]:.2f}"
                )
            else:
                val_loss[epoch], val_accuracy[epoch] = test(
                    test_dataloader, model, criterion
                )
                print(
                    f"\tTest  \tloss: {val_loss[epoch]:.4f},  accuracy: {100 * val_accuracy[epoch]:.2f}"
                )

        mlflow.log_metric("train_loss", train_loss)
        mlflow.log_metric("val_loss", val_loss)
        mlflow.log_metric("train_accuracy", train_accuracy)
        mlflow.log_metric("val_accuracy", val_accuracy)

    # Метрики
    test_metrics(test_dataloader, model)

    # Сохраняем модель
    torch.save(model.state_dict(), "../models/EfficientNetV2-S.pth")


if __name__ == "__main__":
    main()
