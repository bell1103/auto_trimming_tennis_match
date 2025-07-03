import os
import subprocess

VIDEOS_DIR = "vidoes"
DATASET_DIR = "dataset"

video_files = [f for f in os.listdir(VIDEOS_DIR) if f.endswith(".mp4")]

# Filter videos starting from match6
filtered_videos = []
for f in video_files:
    name = os.path.splitext(f)[0]  # e.g., 'match6'
    # Extract the number after 'match'
    if name.startswith("match"):
        try:
            num = int(name[5:])
            if num >= 6:
                filtered_videos.append(f)
        except ValueError:
            pass  # ignore files that don't have a number after 'match'

for video_file in filtered_videos:
    match_name = os.path.splitext(video_file)[0]
    video_path = os.path.join(VIDEOS_DIR, video_file)
    export_dir = os.path.join(DATASET_DIR, match_name)
    
    print(f"Processing {video_file}...")
    
    os.makedirs(export_dir, exist_ok=True)
    
    subprocess.run([
        "python", "FrameGenerator.py",
        "--video_dir", video_path,
        "--export_dir", export_dir
    ])

print("All videos processed from match6 onward.")
