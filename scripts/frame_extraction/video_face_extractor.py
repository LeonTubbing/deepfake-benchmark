import cv2
import os
from mtcnn.mtcnn import MTCNN
from tqdm import tqdm
import time

def process_video(video_path, output_dir, frame_rate=1):
    """
    Processes a single video:
      - Extracts one frame per second (or at a custom interval)
      - Detects and crops the face using MTCNN
      - Saves the cropped face image with a naming convention based on the video name and frame count.
    """
    os.makedirs(output_dir, exist_ok=True)
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(fps * frame_rate)
    detector = MTCNN()
    frame_count = 0
    saved_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_count % frame_interval == 0:
            detections = detector.detect_faces(frame)
            if detections:
                face = max(detections, key=lambda d: d['box'][2] * d['box'][3])
                x, y, w, h = face['box']
                x, y = max(x, 0), max(y, 0)
                face_crop = frame[y:y+h, x:x+w]
                output_filename = f"{video_name}_P{saved_count+1:03d}.png"
                cv2.imwrite(os.path.join(output_dir, output_filename), face_crop)
                saved_count += 1
        
        frame_count += 1

    cap.release()
    print(f"Processed video '{video_name}': extracted {saved_count} face images.")

# Example usage: Process all videos in a directory with progress bar and time
video_dir = 'give the pathname to your directory'
output_folder = 'give the pathname to your directory'

os.makedirs(output_folder, exist_ok=True)
video_files = [f for f in os.listdir(video_dir) if f.endswith(('.mp4', '.avi'))]

start_time = time.time()

for video_file in tqdm(video_files, desc="Processing videos", unit="video"):
    video_path = os.path.join(video_dir, video_file)
    process_video(video_path, output_folder, frame_rate=1)

elapsed_time = time.time() - start_time
print(f"\n All videos processed in {elapsed_time / 60:.2f} minutes.")