import av 
import numpy as np
import torchvision.transforms as transforms
import os
import json
import torch
from transformers import VivitImageProcessor, VivitModel
from moviepy.editor import VideoFileClip
from huggingface_hub import hf_hub_download

device = torch.device("cpu")

def dummy_process_video(video_name, frame_count=16, resize_shape=(224, 224)):

    try:
        print("video_name", video_name)
        # video_name = video_name.encode('utf-8').strip()
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
        raise ValueError("Video does not have enough frames")

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

def dummy_video_processing():
    data_folder = "dummy_data/videos"

    training_tensors, training_labels = [], []
    validation_tensors, validation_labels = [], []
    test_tensors, test_labels = [], []

    label_index = 0

    for video_name in os.listdir(data_folder):
        video_path = os.path.join(data_folder, video_name)
        video_tensor = dummy_process_video(video_path)

        if video_tensor is None:
            print("Skipping video:", video_name)
            continue

        assert video_tensor.shape[0] == 16
        assert video_tensor.shape[1] == 3
        assert video_tensor.shape[2] == 224
        assert video_tensor.shape[3] == 224

        video_tensor = video_tensor.float()

        if video_name == "69547.mp4":
            validation_tensors.append(video_tensor)
            validation_labels.append(0)
        else:
            training_tensors.append(video_tensor)
            training_labels.append(label_index)
        
        label_index += 1
    
    return training_tensors, training_labels, validation_tensors, validation_labels, test_tensors, test_labels


training_tensors, training_labels, validation_tensors, validation_labels, test_tensors, test_labels = dummy_video_processing()
print(max(training_labels), max(validation_labels))

# turn the arrays into tensors
training_tensors = torch.stack(training_tensors)
training_labels = torch.tensor(training_labels)
validation_tensors = torch.stack(validation_tensors)
validation_labels = torch.tensor(validation_labels)
# test_tensors = torch.stack(test_tensors)
# test_labels = torch.tensor(test_labels)

# print dimensions of tensors
print("training_tensors", training_tensors.shape)
print("training_labels", training_labels.shape)
print("validation_tensors", validation_tensors.shape)
print("validation_labels", validation_labels.shape)
# print("test_tensors", test_tensors.shape)
# print("test_labels", test_labels.shape)

# print("total processed:", total_processed)
# print("total skipped:", total_skipped)

torch.save(training_tensors, "data_tensors/training_tensors.pt")
torch.save(training_labels, "data_tensors/training_labels.pt")
torch.save(validation_tensors, "data_tensors/validation_tensors.pt")
torch.save(validation_labels, "data_tensors/validation_labels.pt")
torch.save(test_tensors, "data_tensors/test_tensors.pt")
torch.save(test_labels, "data_tensors/test_labels.pt")

