import torch
import torch.nn.functional as F

def evaluate_model(model, dataloader, device='cpu', label_map=None, topk=1, verbose=False):
    """
    Evaluates the input model on a given dataloader.

    Args:
        model (nn.Module): The model to evaluate.
        dataloader (DataLoader): DataLoader for the test set.
        device (str): Device to run inference on (mostly cpu).
        label_map (list or dict, optional): For printing predicted character labels.
        topk (int): If > 1, returns top-k accuracy as well.
        verbose (bool): If True, prints per-sample predictions.

    Returns:
        accuracy (float): Top-1 accuracy.
        topk_accuracy (float, optional): Top-k accuracy if topk > 1.
    """

    model.to(device)
    model.eval()

    correct = 0
    total = 0
    topk_correct = 0

    with torch.no_grad():
        for i, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)
            outputs = model(X)

            # Top-1 accuracy
            preds = outputs.argmax(dim=1)
            correct += (preds == y).sum().item()

            # Top-k accuracy
            if topk > 1:
                topk_preds = torch.topk(outputs, topk, dim=1).indices
                match = topk_preds.eq(y.view(-1, 1).expand_as(topk_preds))
                topk_correct += match.any(dim=1).sum().item()

            if verbose:
                for idx in range(len(X)):
                    pred_label = preds[idx].item()
                    true_label = y[idx].item()
                    pred_char = label_map[pred_label] if label_map else str(pred_label)
                    true_char = label_map[true_label] if label_map else str(true_label)
                    print(f"[{i * len(X) + idx}] True: {true_char} | Predicted: {pred_char}")

            total += y.size(0)

    top1_acc = 100 * correct / total
    if topk > 1:
        topk_acc = 100 * topk_correct / total
        return top1_acc, topk_acc
    return top1_acc