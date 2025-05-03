import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import random
import numpy as np

class TorchCNN(nn.Module):
    def __init__(self, input_dim, num_classes, dropout_rate=0.5, num_filters=100, kernel_size=3, random_state=42):
        super().__init__()
        self.conv = nn.Conv1d(in_channels=input_dim, out_channels=num_filters, kernel_size=kernel_size)
        self.relu = nn.ReLU()
        self.global_pool = nn.AdaptiveMaxPool1d(1)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(num_filters, num_classes)

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA no disponible")
        self.device = torch.device("cuda")
        self.to(self.device)

        self.g = self.set_seed_and_generator(random_state)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # (batch, embedding_dim, seq_len)
        x = self.conv(x)
        x = self.relu(x)
        x = self.global_pool(x).squeeze(2)  # (batch, num_filters)
        x = self.dropout(x)
        return self.fc(x)

    def fit(self, x_train, y_train, x_val, y_val, batch_size=32, epochs=100, patience=5, lr=1e-3):
        x_train, y_train, x_val, y_val = self.np2torch(x_train, y_train, x_val, y_val)
        train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True, generator=self.g)

        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        best_loss = float('inf')
        best_state = None
        patience_counter = 0

        for _ in range(epochs):
            self.train()
            for xb, yb in train_loader:
                optimizer.zero_grad()
                preds = self(xb)
                loss = criterion(preds, yb)
                loss.backward()
                optimizer.step()

            val_loss = self.evaluate(x_val, y_val, criterion)
            if val_loss < best_loss:
                best_loss = val_loss
                best_state = self.state_dict()
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break

        self.load_state_dict(best_state)

    def predict(self, X):
        self.eval()
        X = torch.tensor(X, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            logits = self(X)
            return torch.argmax(logits, dim=1).cpu().numpy()

    def evaluate(self, x_val, y_val, criterion):
        self.eval()
        with torch.no_grad():
            val_preds = self(x_val)
            val_loss = criterion(val_preds, y_val).item()
        return val_loss

    def np2torch(self, x_train, y_train, x_val, y_val):
        x_train_torch = torch.tensor(x_train, dtype=torch.float32).to(self.device)
        y_train_torch = torch.tensor(y_train, dtype=torch.long).to(self.device)
        x_val_torch = torch.tensor(x_val, dtype=torch.float32).to(self.device)
        y_val_torch = torch.tensor(y_val, dtype=torch.long).to(self.device)
        return x_train_torch, y_train_torch, x_val_torch, y_val_torch

    def set_seed_and_generator(self, seed=42):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        g = torch.Generator()
        g.manual_seed(seed)
        return g
