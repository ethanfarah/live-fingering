import os
import torch
import torch.nn as nn

def get_tensors():
    training_tensors = torch.load("data_tensors1/training_tensors.pt")
    training_labels = torch.load("data_tensors1/training_labels.pt")
    validation_tensors = torch.load("data_tensors1/validation_tensors.pt")
    validation_labels = torch.load("data_tensors1/validation_labels.pt")
    test_tensors = torch.load("data_tensors1/test_tensors.pt")
    test_labels = torch.load("data_tensors1/test_labels.pt")

    for i in range(2, 6):
        training_tensors = torch.cat((training_tensors, torch.load(f"data_tensors{i}/training_tensors.pt")))
        training_labels = torch.cat((training_labels, torch.load(f"data_tensors{i}/training_labels.pt")))
        validation_tensors = torch.cat((validation_tensors, torch.load(f"data_tensors{i}/validation_tensors.pt")))
        validation_labels = torch.cat((validation_labels, torch.load(f"data_tensors{i}/validation_labels.pt")))
        test_tensors = torch.cat((test_tensors, torch.load(f"data_tensors{i}/test_tensors.pt")))
        test_labels = torch.cat((test_labels, torch.load(f"data_tensors{i}/test_labels.pt")))
                                
    return training_tensors, training_labels, validation_tensors, validation_labels, test_tensors, test_labels

training_tensors, training_labels, validation_tensors, validation_labels, test_tensors, test_labels = get_tensors()

target_folder = "full_tensors"

# save the tensors
torch.save(training_tensors, target_folder + "/training_tensors.pt")
torch.save(training_labels, target_folder + "/training_labels.pt")
torch.save(validation_tensors, target_folder + "/validation_tensors.pt")
torch.save(validation_labels, target_folder + "/validation_labels.pt")
torch.save(test_tensors, target_folder + "/test_tensors.pt")
torch.save(test_labels, target_folder + "/test_labels.pt")

# print dimensions of tensors
print("training_tensors", training_tensors.shape)
print("training_labels", training_labels.shape)
print("validation_tensors", validation_tensors.shape)
print("validation_labels", validation_labels.shape)
print("test_tensors", test_tensors.shape)
print("test_labels", test_labels.shape)

