import argparse
import os
from Predict import run_inference
from player_detector import run_player_detection
from Predict_bounce import detect_bounces_pipeline
from point_in_play_detection import detect_points_in_play
from extract_play_segments_video import extract_segments_from_video
import json
import cv2


def main(video_path, output_dir, model_path, display_trail=False):
    os.makedirs(output_dir, exist_ok=True)

    # Get name of video 
    video_basename = os.path.splitext(os.path.basename(video_path))[0]

    player_json_out = os.path.join(output_dir, f"{video_basename}_players.json")
    ball_json_out = os.path.join(output_dir, f"{video_basename}_ball.json")
    bounce_output_path = os.path.join(output_dir, f"{video_basename}_bounces.json")
    play_segments_json = os.path.join(output_dir, f"{video_basename}_segments.json")
    shortened_video_path = os.path.join(output_dir, f"{video_basename}_shortened.mp4")
   


    #  Detect players
    _, _ = run_player_detection(video_path, player_json_out)
    print("Player detection complete.")

    #  Detect ball
    _, _, _ = run_inference(video_path, model_path, display_trail)
    print("Ball detection complete.")

    combined_output_path = os.path.join(output_dir, "combined_video.mp4")

    #  Load keypoints
    with open('path_to_court_keypoints.json', 'r') as f:
        court_keypoints = json.load(f)


    # Detect bounces
    with open(ball_json_out, "r") as f:
        ball_data = json.load(f)

    
    detect_bounces_pipeline(ball_data, visualize=False, save_path=bounce_output_path)
    print("Bounce detection complete.")

    # Load bounce data
    with open(bounce_output_path, "r") as f:
        bounces = json.load(f)  

    # Detect in-play points

    points = detect_points_in_play(
        ball_data=ball_data,
        player_boxes=player_json_out,
        bounces=bounces,
        court_keypoints=court_keypoints,
        max_missing=50,
        output_json=play_segments_json
    )
    print(f"[INFO] Detected {len(points)} points.")

    # Extract point segments
    shortened_video_path = os.path.join(output_dir, shortened_video_path)
    extract_segments_from_video(video_path, points, shortened_video_path)
    print(f"[INFO] Saved shortened video to: {shortened_video_path}")


    # #  Generate final combined overlay video
    # combined_output_path = os.path.join(output_dir, "combined_overlay.mp4")
    # overlay_combined_video(video_path, player_json_out, ball_json_out, court_keypoints, combined_output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Full pipeline: Player, Ball, Court detection and overlay")

    parser.add_argument("--video_path", required=True, type=str, help="Path to input video file.")
    parser.add_argument("--output_dir", required=True, type=str, help="Directory to save outputs.")
    parser.add_argument("--model_path", required=True, type=str, help="Path to ball detection model weights.")
    parser.add_argument("--display_trail", default=False, action="store_true", help="Show ball trajectory trail.")

    args = parser.parse_args()

    main(args.video_path, args.output_dir, args.model_path, args.display_trail)
