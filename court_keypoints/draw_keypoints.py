import supervision as sv
import numpy as np


# not using this
class CourtKeypointsDrawer:
    def __init__(self):
        self.keypoint_color = "#ff2c2c"
    
    def draw(self, frames, court_keypoints):
        vertex_annotator = sv.VertexAnnotator(
            color = sv.Color.from_hex(self.keypoint_color),
            radius=8
        )
        vertex_label_annotator = sv.VertexLabelAnnotator(
            color=sv.Color.from_hex(self.keypoint_color),
            text_color=sv.Color.WHITE,
            text_scale=0.5,
            text_thickness=1
        )

        output_frames=[]
        for index,frame in enumerate(frames):
            annotate_frame = frame.copy()

            # court_keypoints[index] is expected to be a list of [x, y] pairs
            keypoints_array = np.array(court_keypoints[index], dtype=np.float32)  # shape: (14, 2)

            # Make sure the array is exactly 2D with shape (N, 2)
            if keypoints_array.ndim != 2 or keypoints_array.shape[1] != 2:
                raise ValueError(f"Each keypoints_array must be of shape (N, 2). Got {keypoints_array.shape}")

            keypoints = sv.KeyPoints(xy=keypoints_array[np.newaxis, ...])  # this is the expected format now

            annotate_frame = vertex_annotator.annotate(scene=annotate_frame, key_points=keypoints)

            annotate_frame = vertex_label_annotator.annotate(scene=annotate_frame, key_points=keypoints)
            output_frames.append(annotate_frame)

        return output_frames