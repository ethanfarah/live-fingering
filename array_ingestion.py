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
                # new tensor dimensions: 1 x F x 2 x 21 x 3
                video_tensor = video_tensor[0]
                # video_tensor = video_tensor.flatten()

                if dataset_class == "train":
                    training_tensors.append(video_tensor)
                    training_labels.append(label)
                elif dataset_class == "val":
                    validation_tensors.append(video_tensor)
                    validation_labels.append(label)
                elif dataset_class == "test":
                    test_tensors.append(video_tensor)
                    test_labels.append(label)
                else:
                    print("Unknown dataset class:", dataset_class)
                    continue

# Convert lists of tensors to single tensors
# train_tensor = torch.cat(training_tensors, dim=0)
# val_tensor = torch.cat(validation_tensors, dim=0)
# test_tensor = torch.cat(test_tensors, dim=0)

# Save the tensors and labels to disk
torch.save((training_tensors, training_labels), os.path.join(tensor_directory, "training_tensors.pt"))
torch.save((validation_tensors, validation_labels), os.path.join(tensor_directory, "validation_tensors.pt"))
torch.save((test_tensors, test_labels), os.path.join(tensor_directory, "test_tensors.pt"))

print("Training set size:", len(training_tensors))
print("Validation set size:", len(validation_tensors))
print("Test set size:", len(test_tensors))
