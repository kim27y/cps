from torch import nn
import torch
def batch_to_device(X, y=None,device="cuda"):
    X_ = []
    for i in X:
        X_.append(i.to(device))
    if torch.is_tensor(y):
        return torch.stack(X_).transpose(1, 0), y.to(device).float()
    else:
        return torch.stack(X_).transpose(1, 0)

def count_dm_params(model):
    trainable_count = 0
    total_count = 0
    if isinstance(model, torch.nn.Sequential):
        for index in model._modules:
            trainable_count += sum(p.numel() for p in model._modules[index].parameters() if p.requires_grad)
            total_count += sum(p.numel() for p in model._modules[index].parameters())
    else:
        total_count = sum(p.numel() for p in model.parameters())
        trainable_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_count, trainable_count, total_count - trainable_count


def print_kdd(product, ton):
    if product == "carrier":
        sub = ton - 750
        print(f"6/{ton}/8/1/{680 + sub}/1/5")
    if product == "e41g":
        print(f"6/{ton}/6/15")
    if product == "pivot":
        print(f"6/{ton}/10/로냉")
    if product == "scm":
        sub = ton - 775
        print(f"5/{ton}/7/5/{710 + sub}/3/15")
    if product == "tes1":
        sub = ton - 750
        print(f"6/{ton}/8/1/{710 + sub}/2/15")
    if product == "um":
        sub = ton - 750
        print(f"6/{ton}/8/1/{710 + sub}/2/15")
    if product == "yoke":
        print(f"5/{ton}/5/15")

def get_criterion(name):
    if name == "bce":
        return torch.nn.BCELoss(reduction="mean")
    elif name == "bce_logits":
        return nn.BCEWithLogitsLoss(reduction="mean")
    elif name == "ce":
        return nn.CrossEntropyLoss(reduction="mean")
    elif name == "mse":
        return nn.MSELoss(reduction="mean")
    else:
        raise NotImplementedError