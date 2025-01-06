import torch
from torchmetrics import AUROC, Accuracy, AveragePrecision, F1Score, Precision, Recall

torch.manual_seed(123)


def test_metrics(dataloader, model, task="multiclass", num_classes=23):
    model.eval()
    DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

    # микро-усреднение
    metrics_accuracy_micro = Accuracy(
        task=task, num_classes=num_classes, average="micro"
    ).to(DEVICE)
    metrics_precision_micro = Precision(
        task=task, num_classes=num_classes, average="micro"
    ).to(DEVICE)
    metrics_recall_micro = Recall(
        task=task, num_classes=num_classes, average="micro"
    ).to(DEVICE)
    metrics_f1score_micro = F1Score(
        task=task, num_classes=num_classes, average="micro"
    ).to(DEVICE)

    # макро-усреднение
    metrics_accuracy_macro = Accuracy(
        task=task, num_classes=num_classes, average="macro"
    ).to(DEVICE)
    metrics_precision_macro = Precision(
        task=task, num_classes=num_classes, average="macro"
    ).to(DEVICE)
    metrics_recall_macro = Recall(
        task=task, num_classes=num_classes, average="macro"
    ).to(DEVICE)
    metrics_f1score_macro = F1Score(
        task=task, num_classes=num_classes, average="macro"
    ).to(DEVICE)
    metrics_aucroc_macro = AUROC(
        task=task, num_classes=num_classes, average="macro"
    ).to(DEVICE)
    metric_aucpr_macro = AveragePrecision(
        task=task, num_classes=num_classes, average="macro"
    ).to(DEVICE)

    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            predictions = model(X_batch)

            # микро-усреднение
            accuracy_micro = metrics_accuracy_micro(predictions, y_batch)
            precision_micro = metrics_precision_micro(predictions, y_batch)
            recall_micro = metrics_recall_micro(predictions, y_batch)
            f1score_micro = metrics_f1score_micro(predictions, y_batch)

            # макро-усреднение
            accuracy_mаcro = metrics_accuracy_macro(predictions, y_batch)
            aucroc_mаcro = metrics_aucroc_macro(predictions, y_batch.int())
            precision_mаcro = metrics_precision_macro(predictions, y_batch)
            recall_mаcro = metrics_recall_macro(predictions, y_batch)
            f1score_macro = metrics_f1score_macro(predictions, y_batch)
            aucpr_macro = metric_aucpr_macro(predictions, y_batch.int())

        # микро-усреднение
        accuracy_micro = metrics_accuracy_micro.compute()
        precision_micro = metrics_precision_micro.compute()
        recall_micro = metrics_recall_micro.compute()
        f1score_micro = metrics_f1score_micro.compute()

        # макро-усреднение
        accuracy_macro = metrics_accuracy_macro.compute()
        aucroc_macro = metrics_aucroc_macro.compute()
        precision_macro = metrics_precision_macro.compute()
        recall_macro = metrics_recall_macro.compute()
        f1score_macro = metrics_f1score_macro.compute()
        aucpr_macro = metric_aucpr_macro.compute()

        print(f"Result metrics for test dataset")
        # микро-усреднение
        print(f"\n>>> Average - MICRO")
        print(f"\tAccuracy:  {accuracy_micro:.2f}")
        print(f"\tPrecision: {precision_micro:.2f}")
        print(f"\tRecall:    {recall_micro:.2f}")
        print(f"\tF1 Score:  {f1score_micro:.2f}")

        # макро-усреднение
        print(f"\n>>> Average - MACRO")
        print(f"\tAccuracy:  {accuracy_macro:.2f}")
        print(f"\tPrecision: {precision_macro:.2f}")
        print(f"\tRecall:    {recall_macro:.2f}")
        print(f"\tF1 Score:  {f1score_macro:.2f}")
        print(f"\tAUC-ROC:   {aucroc_macro:.2f}")
        print(f"\tAUC-PR:    {aucpr_macro:.2f}")
