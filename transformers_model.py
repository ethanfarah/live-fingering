import torch
from torch.utils.data import TensorDataset, DataLoader
from torch import nn
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
import os

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

train_tensors = torch.load("data_tensors/training_tensors.pt")
train_labels = torch.load("data_tensors/training_labels.pt")
val_tensors = torch.load("data_tensors/validation_tensors.pt")
val_labels = torch.load("data_tensors/validation_labels.pt")
test_tensors = torch.load("data_tensors/test_tensors.pt")
test_labels = torch.load("data_tensors/test_labels.pt")

train_dataset = TensorDataset(train_tensors, train_labels)
val_dataset = TensorDataset(val_tensors, val_labels)
test_dataset = TensorDataset(test_tensors, test_labels)

batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

train_labels = train_labels.squeeze()
print("training set shape", train_tensors.size())
print("training label shape", train_labels.size())

class TransformerModel(nn.Module):
    def __init__(self, input_dim, num_heads, num_layers, num_classes):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Linear(input_dim, input_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=input_dim, nhead=num_heads, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pooling = nn.AdaptiveAvgPool1d(1)  # Adaptive pooling to single descriptor
        self.classifier = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer_encoder(x)
        x = x.transpose(1, 2) 
        x = self.pooling(x)
        x = x.flatten(1)
        x = self.classifier(x)
        return x

D = train_tensors.size(2)
number_of_classes = 0
for tensor in [train_labels, val_labels, test_labels]:
    number_of_classes = max(number_of_classes, tensor.max().item() + 1)

model = TransformerModel(input_dim=D, num_heads=7, num_layers=3, num_classes=number_of_classes)
model = model.to(device)

optimizer = Adam(model.parameters(), lr=0.001)
loss_fn = CrossEntropyLoss()

def train_epoch(model, data_loader, loss_fn, optimizer, device):
    model.train()
    total_loss = 0
    for X_batch, y_batch in data_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()

        # print("X_batch and y_batch shapes", X_batch.shape, y_batch.shape)
        output = model(X_batch)

        # print("first output shape", output.shape)
        output = output.view(-1, output.shape[-1])  # Flatten output for loss calculation
        y_batch = y_batch.view(-1)  # Ensure labels are flattened as well

        # print("SHAPES", output.shape, y_batch.shape)
        loss = loss_fn(output, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(data_loader)

def eval_model(model, data_loader, loss_fn, device):
    model.eval()
    total_loss = 0
    total_correct = 0
    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            output = model(X_batch)
            output = output.view(-1, output.shape[-1])  # Flatten output for loss calculation
            y_batch = y_batch.view(-1)  # Ensure labels are flattened as well
            loss = loss_fn(output, y_batch)
            total_loss += loss.item()
            total_correct += (output.argmax(1) == y_batch).type(torch.float).sum().item()
    return total_loss / len(data_loader), total_correct / (len(data_loader.dataset) * data_loader.dataset[0][0].size(0))

n_epochs = 10
for epoch in range(n_epochs):
    print(f"Epoch {epoch + 1}")
    train_loss = train_epoch(model, train_loader, loss_fn, optimizer, device)
    val_loss, val_acc = eval_model(model, val_loader, loss_fn, device)
    print(f"Epoch {epoch + 1}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, val_acc={val_acc:.4f}")


test_loss, test_acc = eval_model(model, test_loader, loss_fn, device)
print(f"Test loss: {test_loss:.4f}, test accuracy: {test_acc:.4f}")