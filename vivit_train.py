import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from torch.optim.lr_scheduler import StepLR
from transformers import VivitConfig, VivitForVideoClassification
from torch.optim.lr_scheduler import StepLR
import gc

def clear_gpu_memory():
    print("Clearing GPU memory...")
    torch.cuda.empty_cache()
    gc.collect()

clear_gpu_memory()

# Load the tensors
training_tensors = torch.load("data_tensors/training_tensors.pt")
training_labels = torch.load("data_tensors/training_labels.pt")
validation_tensors = torch.load("data_tensors/validation_tensors.pt")
validation_labels = torch.load("data_tensors/validation_labels.pt")
test_tensors = torch.load("data_tensors/test_tensors.pt")
test_labels = torch.load("data_tensors/test_labels.pt")

# print max training label value
print("Max training label value:", torch.max(training_labels))
print("Min training label value:", torch.min(training_labels))
print("num classes:", len(torch.unique(training_labels)))

# Adjust the tensor shapes
# training_tensors = training_tensors.permute(0, 2, 1, 3, 4)
# validation_tensors = validation_tensors.permute(0, 2, 1, 3, 4)
# test_tensors = test_tensors.permute(0, 2, 1, 3, 4)

print("First training tensor", training_tensors[0][0])
print("Second training tensor", training_tensors[1][0])
print("Third training tensor", training_tensors[2][0])
print("Fourth training tensor", training_tensors[3][0])

print("training_tensors", training_tensors.shape)
print("training_labels", training_labels.shape)
print("validation_tensors", validation_tensors.shape)

one_third_length = len(training_tensors) // 3 - 1

# Re-create the datasets with the updated tensor shapes
train_dataset = TensorDataset(training_tensors[:one_third_length], training_labels[:one_third_length])
val_dataset = TensorDataset(validation_tensors, validation_labels)
test_dataset = TensorDataset(test_tensors, test_labels)

# DataLoader remains unchanged
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# Load the model
configuration = VivitConfig(
    num_labels=len(torch.unique(training_labels)),
    num_frames=16,
    frame_height=224,
    frame_width=224,
    num_channels=3
)

model = VivitForVideoClassification(configuration)

# Set the device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
print("Using device:", device)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)
scheduler = StepLR(optimizer, step_size=1, gamma=0.1)

# Function to train the model
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=5, learning_rate=0.001):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            # print("INPUTS", inputs.shape)
            outputs = model(inputs).logits
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # Calculate training accuracy
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            if i % 20 == 19:
                print(f"[{epoch + 1}, {i + 1}] loss: {running_loss / 20:.4f}")
                print(f"Training Accuracy: {100 * correct / total:.2f}%")
                running_loss = 0.0
            else:
                print(f"Training Accuracy: {100 * correct / total:.2f}%")

            print("Finished data iteration " + str(i) + " out of " + str(len(train_loader)))

        scheduler.step()

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data in val_loader:
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs).logits
                index, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                # print out the first 100 predicted labels
                if total <= 100:
                    print("Value of predicted index:", index)
                    print("Predicted:", predicted)
                    print("Actual:", labels)

        print(f"Validation Accuracy after epoch {epoch + 1}: {100 * correct / total:.2f}%")
        print("Finished epoch " + str(epoch))


def preval(model, val_loader):
    print("model on validation set without training")
    total = 0
    model.eval()
    correct = 0
    with torch.no_grad():
        for data in val_loader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs).logits
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # print out the first 100 predicted labels
            if total <= 100:
                print("Predicted:", predicted)
                print("Actual:", labels)

print("Starting training")

# Train the model and print training accuracy
train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=5)

# Save the model state
torch.save(model.state_dict(), "model.pth")
