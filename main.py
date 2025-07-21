import argparse
import os
from Predict import run_inference
from player_detector import run_player_detection
from court_keypoints.label_keypoints import LabelKeypoints
from Predict_bounce import detect_bounces_pipeline
import json
import cv2

def draw_players(frame, players):
    for i, bbox in enumerate(players):
        x1, y1, x2, y2 = map(int, bbox)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"Player {i}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    return frame


def draw_ball(frame, ball_pos):
    x, y = ball_pos
    if (x, y) != (0, 0):
        cv2.circle(frame, (x, y), 8, (0, 0, 255), -1)
    return frame

def draw_court_keypoints(frame, keypoints):
    # keypoints is a list of (x,y) tuples
    for idx, point in enumerate(keypoints):
        x, y = point
        cv2.circle(frame, (int(x), int(y)), 5, (255, 0, 0), -1)
        cv2.putText(frame, f"KP {idx}", (int(x)+5, int(y)-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    return frame


def overlay_combined_video(video_path, players_json, ball_json, court_keypoints, output_path):
    with open(players_json, "r") as f:
        players_data = json.load(f)

    with open(ball_json, "r") as f:
        ball_data = json.load(f)

    ball_dict = {frame: (x, y) for frame, x, y in ball_data}

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_key = str(frame_idx)
        if frame_key in players_data:
            frame = draw_players(frame, players_data[frame_key])
        else:
            frame = draw_players(frame, [])  # no players for this frame

        if frame_idx in ball_dict:
            frame = draw_ball(frame, ball_dict[frame_idx])
        if court_keypoints:
            frame = draw_court_keypoints(frame, court_keypoints)

        cv2.putText(frame, f"Frame: {frame_idx}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        out.write(frame)
        frame_idx += 1

    cap.release()
    out.release()
    print(f"[INFO] Saved combined overlay video to: {output_path}")


def main(video_path, output_dir, model_path, display_trail=False):
    os.makedirs(output_dir, exist_ok=True)

    #  Detect players
    player_json_out, _ = run_player_detection(video_path, output_dir)
    print("Player detection complete.")

    #  Detect ball
    ball_video_out, ball_json_out, _ = run_inference(video_path, model_path, display_trail)
    print("Ball detection complete.")

    combined_output_path = os.path.join(output_dir, "combined_video.mp4")

    # Detect court keypoints
    keypoint_extractor = LabelKeypoints(video_path=video_path, output_path="output/court_keypoints.json")
    court_keypoints = keypoint_extractor.run(save_to_files=True)
    print("Court keypoints extracted.")

    # Detect bounces
    with open(ball_json_out, "r") as f:
        ball_data = json.load(f)

    bounce_output_path = os.path.join(output_dir, "bounces.json")
    detect_bounces_pipeline(ball_data, visualize=False, save_path=bounce_output_path)
    print("Bounce detection complete.")

    #  Generate final combined overlay video
    combined_output_path = os.path.join(output_dir, "combined_overlay.mp4")
    overlay_combined_video(video_path, player_json_out, ball_json_out, court_keypoints, combined_output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Full pipeline: Player, Ball, Court detection and overlay")

    parser.add_argument("--video_path", required=True, type=str, help="Path to input video file.")
    parser.add_argument("--output_dir", required=True, type=str, help="Directory to save outputs.")
    parser.add_argument("--model_path", required=True, type=str, help="Path to ball detection model weights.")
    parser.add_argument("--display_trail", default=False, action="store_true", help="Show ball trajectory trail.")

    args = parser.parse_args()

    main(args.video_path, args.output_dir, args.model_path, args.display_trail)
