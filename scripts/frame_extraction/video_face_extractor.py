import os
import time

import cv2
from mtcnn.mtcnn import MTCNN
from tqdm import tqdm

# ─── Configuration ─────────────────────────────────────────────────────────────
# Directory containing input videos (mp4, avi, etc.)
VIDEO_DIR = 'path/to/videos'
# Directory where cropped face images will be saved
OUTPUT_DIR = 'path/to/output_faces'
# How many frames per second to process (1 = one frame each second)
FRAME_RATE = 1
# ───────────────────────────────────────────────────────────────────────────────

def process_video(video_path, output_dir, frame_rate=1):
    """Extract one frame per `frame_rate` second, detect the largest face, and save crops."""
    os.makedirs(output_dir, exist_ok=True)
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Cannot open video: {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    interval = int(fps * frame_rate)
    detector = MTCNN()
    saved = 0

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Only run detection on every `interval`th frame
        if frame_idx % interval == 0:
            faces = detector.detect_faces(frame)
            if faces:
                # Choose the face with the largest area
                face = max(faces, key=lambda f: f['box'][2] * f['box'][3])
                x, y, w, h = face['box']
                # Ensure coords are within the image
                x, y = max(x, 0), max(y, 0)
                crop = frame[y:y+h, x:x+w]
                filename = f"{video_name}_P{saved+1:03d}.png"
                cv2.imwrite(os.path.join(output_dir, filename), crop)
                saved += 1

        frame_idx += 1

    cap.release()
    print(f"[{video_name}] Extracted {saved} faces.")

if __name__ == '__main__':
    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Gather all video files in the input directory
    videos = [
        f for f in os.listdir(VIDEO_DIR)
        if f.lower().endswith(('.mp4', '.avi', '.mov'))
    ]

    start = time.time()
    for vid in tqdm(videos, desc="Processing videos"):
        process_video(
            video_path=os.path.join(VIDEO_DIR, vid),
            output_dir=OUTPUT_DIR,
            frame_rate=FRAME_RATE
        )
    elapsed = time.time() - start
    print(f"All done in {elapsed/60:.2f} minutes.")
