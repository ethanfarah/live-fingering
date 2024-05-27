import av 
import numpy as np
import torchvision.transforms as transforms
import os
import json
import torch
from transformers import VivitImageProcessor, VivitModel
from huggingface_hub import hf_hub_download
import gc

def clear_gpu_memory():
    print("Clearing GPU memory...")
    torch.cuda.empty_cache()
    gc.collect()

# clear_gpu_memory()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

device = "cpu"

def process_video(video_name, frame_count=16, resize_shape=(224, 224)):
    try:
        container = av.open(video_name)
    except UnicodeDecodeError as e:
        print(f"Failed to process {video_name}: {e}")
        return None
    except Exception as e:
        print(f"An error occurred while opening the file: {e}")
        return None

    stream = container.streams.video[0]
    total_frames = stream.frames
    frame_interval = max(1, total_frames // frame_count)

    if total_frames < frame_count:
        print("not enough frames for video:", video_name)
        return None

    frames = []
    try:
        for idx, frame in enumerate(container.decode(video=0)):
            if idx % frame_interval == 0:
                if len(frames) < frame_count:
                    frame = frame.to_image()
                    frames.append(frame)
                else:
                    break
    except Exception as e:
        print(f"An error occurred while processing the video: {e}")
        return None

    while len(frames) < frame_count:
        # pad the frames by repeating the last frame until we have enough
        frames.append(frames[-1] if frames else Image.new('RGB', resize_shape))

    transform = transforms.Compose([
        transforms.Resize(resize_shape),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    tensors = [transform(frame).to(device) for frame in frames]
    return torch.stack(tensors).to(device)
    
def cash_in(threshold, training_tensors, training_labels, validation_tensors, validation_labels, test_tensors, test_labels):
    # turn the arrays into tensors
    training_tensors = torch.stack(training_tensors)
    training_labels = torch.tensor(training_labels)
    validation_tensors = torch.stack(validation_tensors)
    validation_labels = torch.tensor(validation_labels)
    test_tensors = torch.stack(test_tensors)
    test_labels = torch.tensor(test_labels)

    storage_folder_root = "data_tensors" + str(int(threshold / 2000))

    torch.save(training_tensors, storage_folder_root + "/training_tensors.pt")
    torch.save(training_labels, storage_folder_root + "/training_labels.pt")
    torch.save(validation_tensors, storage_folder_root + "/validation_tensors.pt")
    torch.save(validation_labels, storage_folder_root + "/validation_labels.pt")
    torch.save(test_tensors, storage_folder_root + "/test_tensors.pt")
    torch.save(test_labels, storage_folder_root + "/test_labels.pt")
    
    # print dimensions of tensors
    print("training_tensors", training_tensors.shape)
    print("training_labels", training_labels.shape)
    print("validation_tensors", validation_tensors.shape)
    print("validation_labels", validation_labels.shape)
    print("test_tensors", test_tensors.shape)
    print("test_labels", test_labels.shape)

    # clear_gpu_memory()

    return


def process_all_videos(video_folder):
    print("processing all videos")
    json_directory = "./data"

    # Load missing video IDs
    missing_video_ids = set()
    with open(os.path.join(json_directory, "missing.txt"), "r") as missing_file:
        for line in missing_file:
            missing_video_ids.add(line.strip())

    # Initialize containers for data and labels
    training_tensors, training_labels = [], []
    validation_tensors, validation_labels = [], []
    test_tensors, test_labels = [], []

    label_to_idx = {}
    with open('data/wlasl_class_list.txt', 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 2:
                index, label = parts
                label_to_idx[label] = int(index)

    total_processed = 0
    total_skipped = 0
    threshold = 2000
    
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
                    
                    video_path = os.path.join(video_folder, f"{video_id}.mp4")
                    video_tensor = process_video(video_path)

                    try:
                        # print(video_tensor.shape)
                        video_tensor.shape
                    except AttributeError:
                        print("Could not process video:", video_id, video_tensor)
                    
                    total_processed += 1

                    if total_processed % 1000 == 0:
                        print("processed:", total_processed)
                    
                    if video_tensor is None:
                        print("skipping video:", video_id)
                        total_skipped += 1
                        continue
                    assert video_tensor.shape[0] == 16
                    assert video_tensor.shape[1] == 3
                    assert video_tensor.shape[2] == 224
                    assert video_tensor.shape[3] == 224

                    try:
                        dataset_class = crossfold_data[video_id]["subset"]
                    except KeyError:
                        print("Could not fetch the label for video_id:", video_id)
                        continue
                    
                    video_tensor = video_tensor.float()

                    if dataset_class == "train":
                        training_tensors.append(video_tensor)
                        training_labels.append(label_to_idx[label])
                    elif dataset_class == "val":
                        validation_tensors.append(video_tensor)
                        validation_labels.append(label_to_idx[label])
                    elif dataset_class == "test":
                        test_tensors.append(video_tensor)
                        test_labels.append(label_to_idx[label])

                if total_processed >= threshold:
                    cash_in(threshold, training_tensors, training_labels, validation_tensors, validation_labels, test_tensors, test_labels)

                    training_tensors, training_labels = [], []
                    validation_tensors, validation_labels = [], []
                    test_tensors, test_labels = [], []

                    threshold += 2000


    return training_tensors, training_labels, validation_tensors, validation_labels, test_tensors, test_labels

# start processing the videos
training_tensors, training_labels, validation_tensors, validation_labels, test_tensors, test_labels = process_all_videos("data/videos")

print(max(training_labels), max(validation_labels))

# turn the arrays into tensors
training_tensors = torch.stack(training_tensors)
training_labels = torch.tensor(training_labels)
validation_tensors = torch.stack(validation_tensors)
validation_labels = torch.tensor(validation_labels)
test_tensors = torch.stack(test_tensors)
test_labels = torch.tensor(test_labels)

torch.save(training_tensors, "data_tensors/training_tensors.pt")
torch.save(training_labels, "data_tensors/training_labels.pt")
torch.save(validation_tensors, "data_tensors/validation_tensors.pt")
torch.save(validation_labels, "data_tensors/validation_labels.pt")
torch.save(test_tensors, "data_tensors/test_tensors.pt")
torch.save(test_labels, "data_tensors/test_labels.pt")

# print dimensions of tensors
print("training_tensors", training_tensors.shape)
print("training_labels", training_labels.shape)
print("validation_tensors", validation_tensors.shape)
print("validation_labels", validation_labels.shape)
print("test_tensors", test_tensors.shape)
print("test_labels", test_labels.shape)

print("finished preprocessing videos, happy training :)")