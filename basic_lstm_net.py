import torch
import os
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim

# Check device
USE_GPU = True
USE_MPS = torch.backends.mps.is_available()
dtype = torch.float32

if USE_GPU and torch.cuda.is_available():
    device = torch.device('cuda')
elif USE_MPS:
    device = torch.device('mps')
else:
    device = torch.device('cpu')

print(f'Using device: {device}')

# Load the data from the .pt files
base_dir = os.getcwd()
train_tensors, train_labels = torch.load(os.path.join(base_dir, 'data_tensors', 'training_tensors.pt'))
val_tensors, val_labels = torch.load(os.path.join(base_dir, 'data_tensors', 'validation_tensors.pt'))
test_tensors, test_labels = torch.load(os.path.join(base_dir, 'data_tensors', 'test_tensors.pt'))

# Normalize the data
train_tensors = (train_tensors - train_tensors.min()) / (train_tensors.max() - train_tensors.min())
val_tensors = (val_tensors - val_tensors.min()) / (val_tensors.max() - val_tensors.min())
test_tensors = (test_tensors - test_tensors.min()) / (test_tensors.max() - test_tensors.min())

# Convert labels to long type for CrossEntropyLoss
train_labels = train_labels.long().squeeze()
val_labels = val_labels.long().squeeze()
test_labels = test_labels.long().squeeze()

# Move tensors to the correct device
train_tensors, train_labels = train_tensors.to(device), train_labels.to(device)
val_tensors, val_labels = val_tensors.to(device), val_labels.to(device)
test_tensors, test_labels = test_tensors.to(device), test_labels.to(device)

# Custom Dataset class
class SignLanguageDataset(Dataset):
    def __init__(self, tensors, labels):
        self.tensors = tensors
        self.labels = labels

    def __len__(self):
        return len(self.tensors)

    def __getitem__(self, idx):
        return self.tensors[idx], self.labels[idx]

# Create Dataset and DataLoader
train_dataset = SignLanguageDataset(train_tensors, train_labels)
val_dataset = SignLanguageDataset(val_tensors, val_labels)
test_dataset = SignLanguageDataset(test_tensors, test_labels)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Define the model with a 1D CNN and LSTM
class SignLanguageModel(nn.Module):
    def __init__(self, input_dim, cnn_out_channels, cnn_kernel_size, hidden_dim, output_dim):
        super(SignLanguageModel, self).__init__()
        self.conv1d = nn.Conv1d(input_dim, cnn_out_channels, kernel_size=cnn_kernel_size, stride=1, padding=cnn_kernel_size // 2)
        self.relu = nn.ReLU()
        self.lstm = nn.LSTM(cnn_out_channels, hidden_dim, batch_first=True, bidirectional=True)
        self.fc1 = nn.Linear(hidden_dim * 2, 128)
        self.fc2 = nn.Linear(128, output_dim)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # x: [batch_size, seq_len, input_dim]
        x = x.transpose(1, 2)  # Change to [batch_size, input_dim, seq_len] for Conv1d
        x = self.conv1d(x)  # Apply 1D CNN
        x = self.relu(x)
        x = x.transpose(1, 2)  # Change back to [batch_size, seq_len, cnn_out_channels]
        x, _ = self.lstm(x)
        x = self.relu(x[:, -1, :])  # Use only the last output of the LSTM
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Hyperparameters
input_dim = 126  # Each frame has 126 features
cnn_out_channels = 64  # Number of output channels for Conv1d
cnn_kernel_size = 3  # Kernel size for Conv1d
hidden_dim = 32
output_dim = train_labels.max().item() + 1  # Number of classes

model = SignLanguageModel(input_dim, cnn_out_channels, cnn_kernel_size, hidden_dim, output_dim).to(device)

# Define loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)  # Adjust the learning rate as needed

# Training loop
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=20):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        print(f'Epoch {epoch}/{num_epochs - 1}, Loss: {epoch_loss:.4f}')

        # Validation phase
        model.eval()
        val_loss = 0.0
        corrects = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                corrects += torch.sum(preds == labels.data)

        val_loss = val_loss / len(val_loader.dataset)
        accuracy = corrects.float() / len(val_loader.dataset)  # Change to float() for float32 dtype
        print(f'Validation Loss: {val_loss:.4f}, Accuracy: {accuracy:.4f}')

# Train the model
train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=200)

# Save the model
torch.save(model.state_dict(), 'model_weights.pth')
