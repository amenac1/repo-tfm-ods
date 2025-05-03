import torch
import random
import numpy as np

def np2torch(x_train, y_train, x_val, y_val, len_train, len_val, device):
    x_train_torch = torch.tensor(x_train, dtype=torch.float32).to(device)
    y_train_torch = torch.tensor(y_train, dtype=torch.long).to(device)
    len_train_torch = torch.tensor(len_train, dtype=torch.long).to(device)
    x_val_torch = torch.tensor(x_val, dtype=torch.float32).to(device)
    y_val_torch = torch.tensor(y_val, dtype=torch.long).to(device)
    len_val_torch = torch.tensor(len_val, dtype=torch.long).to(device)

    return x_train_torch, y_train_torch, len_train_torch, x_val_torch, y_val_torch, len_val_torch

def set_seed_and_generator(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    generator = torch.Generator()
    generator.manual_seed(42)
    return generator

def train_epoch(model, dataloader, optimizer, criterion):
    model.train()
    for xb, yb in dataloader:
        optimizer.zero_grad()
        preds = model(xb)
        loss = criterion(preds, yb)
        loss.backward()
        optimizer.step()

def evaluate(model, x_val, y_val, criterion):
    model.eval()
    with torch.no_grad():
        val_preds = model(x_val)
        val_loss = criterion(val_preds, y_val).item()
    return val_loss

def check_early_stopping(val_loss, best_loss, patience_counter, patience, model):
    if val_loss < best_loss:
        return val_loss, model.state_dict(), 0, False
    else:
        patience_counter += 1
        return best_loss, None, patience_counter, patience_counter >= patience
