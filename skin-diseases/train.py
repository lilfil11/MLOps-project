import torch
import numpy as np

from torch import nn
from tqdm import tqdm

from constants import *
from metrics import test_metrics
from dataset import MedicalImageDataset

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset

from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights


def train(dataloader, model, criterion, optimizer, scheduler):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)

    model.train()
    loss_avg, accuracy_avg = 0, 0
    for X_batch, y_batch in tqdm(dataloader):
        X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)

        optimizer.zero_grad()
        predictions = model(X_batch)
        loss = criterion(predictions, y_batch.long())
        accuracy = (predictions.argmax(1) == y_batch).type(torch.float).sum().item()
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

        for X_batch, y_batch in dataloader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)

            predictions = model(X_batch)
            loss_avg += criterion(predictions, y_batch.long()).item()
            accuracy_avg += (predictions.argmax(1) == y_batch).type(torch.float).sum().item()

    return loss_avg / num_batches, accuracy_avg / size


def main():
    # Датасет
    train_dataset = MedicalImageDataset(
        img_dir=DATASET_PATH,
        data='train',
        transform=PREPROCESS
    )
    test_dataset = MedicalImageDataset(
        img_dir=DATASET_PATH,
        data='test',
        transform=PREPROCESS
    )
    if VALIDATION:
        train_idx, val_idx = train_test_split(np.arange(len(train_dataset)), test_size=0.2, shuffle=True,
                                              random_state=0)
        train_dataset = Subset(train_dataset, train_idx)
        val_dataset = Subset(train_dataset, val_idx)
        val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Обучаем модель
    weights = EfficientNet_V2_S_Weights.DEFAULT
    model = efficientnet_v2_s(weights=weights).to(DEVICE)
    model.classifier[1] = nn.Linear(1280, 23).to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4, weight_decay=5e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    train_loss, train_accuracy = (np.zeros(NUM_EPOCHS), np.zeros(NUM_EPOCHS))
    val_loss, val_accuracy = (np.zeros(NUM_EPOCHS), np.zeros(NUM_EPOCHS))

    for epoch in range(NUM_EPOCHS):
        train_loss[epoch], train_accuracy[epoch] = train(train_dataloader, model, criterion, optimizer, scheduler)
        print(f'\n>>> Epoch {epoch + 1}\n\tTrain\tloss: {train_loss[epoch]:.4f},  accuracy: {100 * train_accuracy[epoch]:.2f}')

        if VALIDATION:
            val_loss[epoch], val_accuracy[epoch] = test(val_dataloader, model, criterion)
            print(f'\tVal  \tloss: {val_loss[epoch]:.4f},  accuracy: {100 * val_accuracy[epoch]:.2f}')
        else:
            val_loss[epoch], val_accuracy[epoch] = test(test_dataloader, model, criterion)
            print(f'\tTest  \tloss: {val_loss[epoch]:.4f},  accuracy: {100 * val_accuracy[epoch]:.2f}')

    # Метрики
    test_metrics(test_dataloader, model)

    # Сохраняем модель
    torch.save(model.state_dict(), '../models/model.pth')


if __name__ == '__main__':
    main()
