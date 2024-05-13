import json
import os
import numpy as np
import torch

json_directory = "./data"
numpy_directory = "./data/landmarks"

# import missing.txt file from /data/missing.txt
missing_video_ids = set()
with open(os.path.join(json_directory, "missing.txt"), "r") as missing_file:
    for line in missing_file:
        missing_video_ids.add(line.strip())

training_tensors = []
training_labels = []

validation_tensors = []
validation_labels = []

test_tensors = []
test_labels = []

with open (os.path.join(json_directory, "WLASL_v0.3.json"), "r") as labeling_set:
    with open (os.path.join(json_directory, "nslt_2000.json"), "r") as crossfold_set:
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
                video_tensor = video_tensor.unsqueeze(0)
    
                dataset_class = ""
                try:
                    dataset_class = crossfold_data[video_id]["subset"]
                except KeyError as error:
                    print("Could not fetch the label for video_id:", video_id)
                    continue
                
                if dataset_class == "train":
                    training_tensors.append(video_tensor)
                    training_labels.append(dataset_class)
                    # print("Added video tensor to training set", video_tensor)
                    # print("label", dataset_class)
                elif dataset_class == "val":
                    validation_tensors.append(video_tensor)
                    validation_labels.append(dataset_class)
                elif dataset_class == "test":
                    test_tensors.append(video_tensor)
                    test_labels.append(dataset_class)
                else:
                    print("Unknown dataset class:", dataset_class)
                    continue

print("Training set size:", len(training_tensors))
print("Validation set size:", len(validation_tensors))
print("Test set size:", len(test_tensors))

# print("First element of training set:", training_tensors[0])
# print("First element of validation set:", validation_tensors[0])
# print("First element of test set:", test_tensors[0])

print("First element of training set label:", training_labels[0])
print("First element of validation set label:", validation_labels[0])
print("First element of test set label:", test_labels[0])

# Save the tensors and labels to disk
torch.save((training_tensors, training_labels), os.path.join(json_directory, "training_tensors_labels.pt"))
torch.save((validation_tensors, validation_labels), os.path.join(json_directory, "validation_tensors_labels.pt"))
torch.save((test_tensors, test_labels), os.path.join(json_directory, "test_tensors_labels.pt"))
