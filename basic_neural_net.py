import torch
import torch.nn as nn
import torch.optim as optim
from array_ingestion import training_tensors, training_labels, validation_tensors, validation_labels, test_tensors, test_labels

# Check for GPU availability
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

print("First data point")
print("Sample data from the tensor1:", len(training_tensors))
print("Sample data from the tensor1:", len(training_tensors[0]))
print("Sample data from the tensor1:", len(training_tensors[0][0]))
print("Sample data from the tensor1:", len(training_tensors[0][0][0]))
print("Sample data from the tensor1:", len(training_tensors[0][0][0][0]))
print("Sample data from the tensor1:", len(training_tensors[0][0][0][0][0]))
# print("Sample data from the tensor1:", len(training_tensors[0][0][0][0][0][0]))

print("Second data point")
print("Sample data from the tensor2:", len(training_tensors))
print("Sample data from the tensor2:", len(training_tensors[1]))
print("Sample data from the tensor2:", len(training_tensors[1][0]))
print("Sample data from the tensor2:", len(training_tensors[1][0][0]))
print("Sample data from the tensor2:", len(training_tensors[1][0][0][0]))
print("Sample data from the tensor2:", len(training_tensors[1][0][0][0][0]))
# print("Sample data from the tensor2:", len(training_tensors[1][0][0][0][0][0]))

class Dataset(torch.utils.data.Dataset):
    def __init__(self, tensors, labels):
        self.tensors = tensors
        self.labels = labels

    def __len__(self):
        return len(self.tensors)

    def __getitem__(self, index):
        tensor = self.tensors[index].to(device)
        label = self.labels[index].to(device)
        return tensor, label

class SimpleNeuralNet(nn.Module):
    def __init__(self):
        super(SimpleNeuralNet, self).__init__()
        self.fc1 = nn.Linear(9450, 512)  # Adjusted input dimensions after flattening
        self.bn1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, 100)

    def forward(self, x):
        x = torch.relu(self.bn1(self.fc1(x)))
        x = torch.relu(self.bn2(self.fc2(x)))
        x = self.fc3(x)
        return x

# Initialize model, send it to GPU
model = SimpleNeuralNet().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

training_set = Dataset(training_tensors, training_labels)
training_loader = torch.utils.data.DataLoader(training_set, batch_size=32, shuffle=True)

# Training loop
model.train()
for epoch in range(10):  # number of epochs can be adjusted
    running_loss = 0.0
    for i, (inputs, labels) in enumerate(training_loader):
        inputs = inputs.view(inputs.shape[0], -1)  # Flatten the tensors
        optimizer.zero_grad()
        outputs = model(inputs.float())
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if (i + 1) % 10 == 0:
            print(f"[{epoch + 1}, {i + 1}] loss: {running_loss / 10}")
            running_loss = 0.0

print("Finished training")

# Validation
model.eval()
correct = 0
total = 0
val_set = Dataset(val_tensor, val_labels)
val_loader = torch.utils.data.DataLoader(val_set, batch_size=32, shuffle=False)

with torch.no_grad():
    for inputs, labels in val_loader:
        inputs = inputs.view(inputs.shape[0], -1)  # Flatten the tensors
        outputs = model(inputs.float())
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Accuracy of the network on the validation set: {100 * correct / total}%")
