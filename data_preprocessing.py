import cv2
import mediapipe as mp
import os
import numpy as np
from multiprocessing import Pool

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,max_num_hands=2, min_detection_confidence=0.5)

def process_video(video_path):
  cap = cv2.VideoCapture(video_path)
  max_hands = 2  # Maximum number of hands you want to detect per frame
  landmark_count = 21  # Number of landmarks per hand in MediaPipe
  video_results = []

  while cap.isOpened():
      success, image = cap.read()
      if not success:
          break

      results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
      
      # Initialize frame data with placeholders for max_hands
      frame_data = np.full((max_hands, landmark_count, 3), -1.0, dtype=float)
      
      if results.multi_hand_landmarks:
          for i, hand_landmarks in enumerate(results.multi_hand_landmarks[:max_hands]):
              for j, lm in enumerate(hand_landmarks.landmark):
                  frame_data[i, j] = [lm.x, lm.y, lm.z]
      
      video_results.append(frame_data)

  cap.release()
  return np.array(video_results)

def process_and_save(video_name, video_folder, output_folder):
  video_path = os.path.join(video_folder, video_name)
  video_results = process_video(video_path)
  npy_path = os.path.join(output_folder, f"{os.path.splitext(video_name)[0]}.npy")
  np.save(npy_path, video_results)
  return f"Processed and saved {video_name}"

def main():
  base_dir = os.getcwd()
  video_folder = os.path.join(base_dir, 'data', 'videos')
  output_folder = os.path.join(base_dir, 'data', 'landmarks')
  video_files = os.listdir(video_folder)

  with Pool(processes=os.cpu_count()) as pool:
      results = pool.starmap(process_and_save, [(video_name, video_folder, output_folder) for video_name in video_files])
      for result in results:
          print(result)

if __name__ == '__main__':
    main()