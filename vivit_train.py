import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from torchvision.models.video import r3d_18

training_tensors = torch.load("data_tensors/training_tensors.pt")
training_labels = torch.load("data_tensors/training_labels.pt")
validation_tensors = torch.load("data_tensors/validation_tensors.pt")
validation_labels = torch.load("data_tensors/validation_labels.pt")
test_tensors = torch.load("data_tensors/test_tensors.pt")
test_labels = torch.load("data_tensors/test_labels.pt")

training_tensors = training_tensors.permute(0, 2, 1, 3, 4)
validation_tensors = validation_tensors.permute(0, 2, 1, 3, 4)

# Re-create the datasets with the updated tensor shapes
train_dataset = TensorDataset(training_tensors, training_labels)
val_dataset = TensorDataset(validation_tensors, validation_labels)

# DataLoader remains unchanged
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

train_dataset = TensorDataset(training_tensors, training_labels)
val_dataset = TensorDataset(validation_tensors, validation_labels)

train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

model = r3d_18(pretrained=True)

num_classes = max(training_labels).item() + 1

model.fc = nn.Linear(model.fc.in_features, num_classes)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
print("using device:", device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 100 == 99:
                print(f"[{epoch + 1}, {i + 1}] loss: {running_loss / 100}")
                running_loss = 0.0

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data in val_loader:
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print(f"Accuracy after epoch {epoch + 1}: {100 * correct / total}")


train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=1)

torch.save(model.state_dict(), "model.pth")
