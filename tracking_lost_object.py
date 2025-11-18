import logging
from ultralytics import YOLO
import cv2
import numpy as np
from boxmot import BotSort
from pathlib import Path
from typing import Optional, Tuple
from utilis.camera_utilis import VideoPlayer, VideoPlayerOffline
from utilis.rules_utilis import TopOfChainOfCommand, DepthMovementDetector, MovementDirectionDetector
import os
import argparse
import openvino as ov
from utilis.draw_chart import decide_tracker_movement, clear_file, plot_tracker_movements
from utilis.tracker_utilis import (preprocess_image, 
                                 postprocess)



os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
core = ov.Core()

class Model:
    def __init__(self, model_path):
        det_model_path = Path(model_path)
        det_ov_model = core.read_model(str(det_model_path))
        try:
            det_ov_model.reshape({0: [1, 3, 640, 640]})
        except Exception:
            pass
        self.det_compiled_model = core.compile_model(det_ov_model, 'AUTO')

    def detect(self, image):
        preprocessed_image = preprocess_image(image)
        result = self.det_compiled_model(preprocessed_image)
        boxes = result[self.det_compiled_model.output(0)]
        input_hw = preprocessed_image.shape[2:]
        detections = postprocess(pred_boxes=boxes, input_hw=input_hw, orig_img=image, classess=None)
        return detections

class FinalMovementAnalyzer:
    def __init__(self, threshold_direction=3, threshold_depth=0.1, threshold_frames=37):
        """
        Initialize the FinalMovementAnalyzer.

        Args:
            threshold_direction (int): Threshold for MovementDirectionDetector to detect movement.
            threshold_depth (float): Threshold for DepthMovementDetector to detect depth change.
            threshold_frames (int): Number of frames required for TopOfChainOfCommand analysis.
        """
        self.threshold_direction = threshold_direction
        self.threshold_depth = threshold_depth
        self.threshold_frames = threshold_frames
        self.moving_objects = set()  # Set of track_ids determined to be moving

    def analyze(self, tracked_objects, current_frame):
        """
        Analyze the movement of tracked objects and log directions.

        Args:
            tracked_objects (dict): Dictionary {track_id: {frame_num: (x1, y1, x2, y2)}} of tracked objects.
            current_frame (int): Current frame number being processed.

        Returns:
            list: List of track_ids that are moving and present in the current frame.
        """
        for track_id, frames_dict in tracked_objects.items():
            frames = sorted(frames_dict.items())
            if len(frames) >= self.threshold_frames + 1 and track_id not in self.moving_objects:
                analysis_frames = frames[:self.threshold_frames + 1]
                toc = TopOfChainOfCommand(analysis_frames)
                dmd = DepthMovementDetector(self.threshold_depth)
                if not toc.is_not_moving_out(self.threshold_frames):
                    self.moving_objects.add(track_id)
                    logging.info(f"Track ID {track_id} is moving (lateral movement detected)")
                elif dmd.is_moving_in_depth(analysis_frames):
                    self.moving_objects.add(track_id)
                    logging.info(f"Track ID {track_id} is moving (depth movement detected)")
            if current_frame % 10 == 0 and len(frames) >= 2:
                num_frames = min(10, len(frames))
                recent_frames = frames[-num_frames:]
                bboxes = [bbox for _, bbox in recent_frames]
                mdd = MovementDirectionDetector(bboxes, self.threshold_direction)
                main_direction = mdd.get_main_direction()
                logging.info(f"Track ID {track_id} at frame {current_frame}: Main direction = {main_direction}")

        # Check for false positives: if center after 80 frames is in bbox from 80 frames ago
        to_remove = set()
        for track_id in self.moving_objects:
            if current_frame in tracked_objects[track_id]:
                frames_dict = tracked_objects[track_id]
                frame_nums = sorted(frames_dict.keys())
                past_frame_nums = [f for f in frame_nums if f <= current_frame - 80]
                if past_frame_nums:
                    past_frame_num = max(past_frame_nums)
                    bbox_past = frames_dict[past_frame_num]
                    bbox_current = frames_dict[current_frame]
                    center_current = TopOfChainOfCommand.bbox_center(bbox_current)
                    if TopOfChainOfCommand.point_in_bbox(center_current, bbox_past):
                        logging.info(f"Track ID {track_id} was a false positive: center still inside bbox from frame {past_frame_num}")
                        to_remove.add(track_id)
        self.moving_objects -= to_remove

        moving_in_current_frame = [
            track_id for track_id in self.moving_objects
            if current_frame in tracked_objects[track_id]
        ]
        return moving_in_current_frame

class AirportItemTracker:
    def __init__(
        self,
        det_path: str,
        reid_weights_path: str,
        video_path: str
    ):
        logging.basicConfig(
            filename='unattended_items_at_airport_update.log',
            level=logging.INFO,
            format='%(asctime)s - %(message)s'
        )
        self.ov_model = Model(det_path)
        self.tracker = BotSort(Path(reid_weights_path), device='cpu', half=False)
        self.video_player = VideoPlayerOffline(video_path)
        self.AIRPORT_ITEM_CLASS_IDS = {
            26: 'handbag',
            27: 'backpack',
            28: 'suitcase'
        }
        self.airport_classes = list(self.AIRPORT_ITEM_CLASS_IDS.keys())
        self.tracked_objects = {}
        self.frame_count = 0
        self.final_analyzer = FinalMovementAnalyzer()

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        self.frame_count += 1
        img = cv2.resize(frame, None, fx=0.5, fy=0.5)
        results_ov = self.ov_model.detect(img)[0]['det']
        
        
        # Check if results_ov is empty or not 2D
        if results_ov.size == 0 or results_ov.ndim != 2:
            # No valid detections; return the frame as is
            cv2.imshow('Tracking', img)
            return img
        
        # Ensure the last column contains class IDs
        if results_ov.shape[1] < 6:
            logging.warning(f"Unexpected detection format: {results_ov.shape}")
            cv2.imshow('Tracking', img)
            return img
        
        # Filter detections for airport item classes
        detections = results_ov[np.isin(results_ov[:, -1], self.airport_classes)]
        if detections.size > 0:
            tracks = self.tracker.update(detections, img)
            for track in tracks:
                if len(track) < 6:
                    continue
                x1, y1, x2, y2, track_id, cls_id = map(int, track[:6])
                bbox = (x1, y1, x2, y2)
                if track_id not in self.tracked_objects:
                    self.tracked_objects[track_id] = {}
                self.tracked_objects[track_id][self.frame_count] = bbox
            moving_ids = self.final_analyzer.analyze(self.tracked_objects, self.frame_count)
            self._draw_tracked_objects(img, tracks, moving_ids)
        else:
            tracks = []
            moving_ids = []
        
        cv2.imshow('Tracking', img)
        return img

    def _draw_tracked_objects(self, img: np.ndarray, tracks: np.ndarray, moving_ids: list):
        for track in tracks:
            if len(track) < 6:
                continue
            x1, y1, x2, y2, track_id, cls_id = map(int, track[:6])
            label = self.AIRPORT_ITEM_CLASS_IDS.get(cls_id, 'unknown')
            color = (0, 255, 0) if track_id in moving_ids else (0, 0, 255)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            text = f"{label} ID:{track_id}"
            cv2.putText(img, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Analyze tracker movement patterns from log files",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--logfile",
        default="unattended_items_at_airport_update.log",
        help="Path to tracker log file"
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Display movement plot"
    )
    args =parser.parse_args()
    
    tracker = AirportItemTracker(
        reid_weights_path='osnet_x0_25_msmt17.pt',
        video_path='/home/arshia/Documents/Git hub/lost_object/videos/#Airport luggage conveyor belt system #short #shortvedio #viral.mp4',
        det_path='./models/yolo11n_openvino_model_int8/yolo11n.xml',
    )
    file_path = args.logfile

    if os.path.exists(file_path) and os.path.isfile(file_path):
        clear_file("./unattended_items_at_airport_update.log")
        print(f"Removed existing file: {file_path}")
    else:
        print(f"File does not exist: {file_path}")

    tracker.video_player.start()
    while True:
        frame = tracker.video_player.next()
        if frame is None:
            break
        processed_frame = tracker.process_frame(frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    tracker.video_player.stop()
    cv2.destroyAllWindows()
    movement_state, track_data = decide_tracker_movement(file_path)
    for tid, state in movement_state.items():
        print(f"Tracker {tid}: {state}")




   