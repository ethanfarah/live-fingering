import json
import os
import numpy as np
import torch

json_directory = "./data"
numpy_directory = "./data/landmarks"
tensor_directory = "./data_tensors"

# Load missing video IDs
missing_video_ids = set()
with open(os.path.join(json_directory, "missing.txt"), "r") as missing_file:
    for line in missing_file:
        missing_video_ids.add(line.strip())

# Initialize containers for data and labels
training_tensors, training_labels = [], []
validation_tensors, validation_labels = [], []
test_tensors, test_labels = [], []

longest_video = 0

# gets the int from the label
label_to_idx = {}
with open('data/wlasl_class_list.txt', 'r') as f:
    for line in f:
        parts = line.strip().split('\t')
        if len(parts) == 2:
            index, label = parts
            label_to_idx[label] = int(index)

# Load dataset info
with open(os.path.join(json_directory, "WLASL_v0.3.json"), "r") as labeling_set:
    with open(os.path.join(json_directory, "nslt_2000.json"), "r") as crossfold_set:
        crossfold_data = json.load(crossfold_set)
        labeling_data = json.load(labeling_set)

        for item in labeling_data:
            label = item["gloss"]
            for video_clips in item["instances"]:
                video_id = video_clips["video_id"]
                if video_id in missing_video_ids:
                    continue

                video_path = os.path.join(numpy_directory, f"{video_id}.npy")
                video_tensor = np.load(video_path)
                video_tensor = torch.from_numpy(video_tensor)
                video_tensor = video_tensor.unsqueeze(0)  # Ensure the tensor has a batch dimension

                try:
                    dataset_class = crossfold_data[video_id]["subset"]
                except KeyError:
                    print("Could not fetch the label for video_id:", video_id)
                    continue
                
                # current tensor dimensions: 1 x F x 2 x 21 x 3
                video_tensor = video_tensor.float()
                # new tensor dimensions: F x 2 x 21 x 3
                video_tensor = video_tensor[0]

                # makes video tensor F x (2*21*3)
                video_tensor = video_tensor.view(video_tensor.size(0), -1)
                assert video_tensor.size(1) == 126, f"Expected 126 features, got {video_tensor.size(1)}"

                if video_tensor.size(0) > longest_video:
                    longest_video = video_tensor.size(0)

                if dataset_class == "train":
                    training_tensors.append(video_tensor.clone().detach())
                    training_labels.append(torch.tensor([label_to_idx[label]]))
                elif dataset_class == "val":
                    validation_tensors.append(video_tensor.clone().detach())
                    validation_labels.append(torch.tensor([label_to_idx[label]]))
                elif dataset_class == "test":
                    test_tensors.append(video_tensor.clone().detach())
                    test_labels.append(torch.tensor([label_to_idx[label]]))
                else:
                    print("Unknown dataset class:", dataset_class)
                    continue

#print size of label_to_index
print(len(label_to_idx))
# print largest value of label_to_index
print(max(label_to_idx.values()))

# pad each tensor to the longest video length with -1
for i in range(len(training_tensors)):
    training_tensors[i] = torch.cat((training_tensors[i], torch.zeros(longest_video - training_tensors[i].size(0), 126) - 1))
for i in range(len(validation_tensors)):
    validation_tensors[i] = torch.cat((validation_tensors[i], torch.zeros(longest_video - validation_tensors[i].size(0), 126) - 1))
for i in range(len(test_tensors)):
    test_tensors[i] = torch.cat((test_tensors[i], torch.zeros(longest_video - test_tensors[i].size(0), 126) - 1))

# turn the arrays into tensors
training_tensors = torch.stack(training_tensors)
training_labels = torch.stack(training_labels)
validation_tensors = torch.stack(validation_tensors)
validation_labels = torch.stack(validation_labels)
test_tensors = torch.stack(test_tensors)
test_labels = torch.stack(test_labels)

# flatten all tensors to N x F x (2*21*3)
training_tensors = training_tensors.view(training_tensors.size(0), training_tensors.size(1), -1)
validation_tensors = validation_tensors.view(validation_tensors.size(0), training_tensors.size(1), -1)
test_tensors = test_tensors.view(test_tensors.size(0), training_tensors.size(1), -1)

# Save the tensors and labels to disk
torch.save(training_tensors, os.path.join(tensor_directory, "training_tensors.pt"))
torch.save(training_labels, os.path.join(tensor_directory, "training_labels.pt"))
torch.save(validation_tensors, os.path.join(tensor_directory, "validation_tensors.pt"))
torch.save(validation_labels, os.path.join(tensor_directory, "validation_labels.pt"))
torch.save(test_tensors, os.path.join(tensor_directory, "test_tensors.pt"))
torch.save(test_labels, os.path.join(tensor_directory, "test_labels.pt"))

print("Training set size:", training_tensors.size())
print("Validation set size:", validation_tensors.size())
print("Test set size:", test_tensors.size())