
from collections import Counter

class TopOfChainOfCommand:
    def __init__(self, frames):
        """
        frames: list of (frame_number, (x1, y1, x2, y2))
        """
        self.frames = frames

    @staticmethod
    def bbox_center(bbox):
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    @staticmethod
    def point_in_bbox(center, bbox):
        x, y = center
        x1, y1, x2, y2 = bbox
        return x1 <= x <= x2 and y1 <= y <= y2

    def is_not_moving_out(self, threshold=25):
        """
        Checks if the object stays inside the first bbox for the next 'threshold' frames.
        Returns:
            True if not moving out (all centers inside), False otherwise.
        """
        if len(self.frames) < threshold + 1:
            raise ValueError("Not enough frames for the given threshold.")
        first_bbox = self.frames[0][1]
        for i in range(1, threshold + 1):
            _, bbox = self.frames[i]
            center = self.bbox_center(bbox)
            if not self.point_in_bbox(center, first_bbox):
                return False
        return True


class DepthMovementDetector:
    def __init__(self, threshold):
        """Initialize the detector with a threshold for size change."""
        self.threshold = threshold

    def is_moving_in_depth(self, frames):
        """Determine if the object is moving in depth based on bounding boxes."""
        if not frames or len(frames) < 2:
            return False
        _, ref_bbox = frames[0]
        ref_x1, ref_y1, ref_x2, ref_y2 = ref_bbox
        ref_center = ((ref_x1 + ref_x2) / 2, (ref_y1 + ref_y2) / 2)
        ref_size = (ref_x2 - ref_x1) * (ref_y2 - ref_y1)
        for frame in frames[1:]:
            _, bbox = frame
            x1, y1, x2, y2 = bbox
            center = ((x1 + x2) / 2, (y1 + y2) / 2)
            if ref_x1 <= center[0] <= ref_x2 and ref_y1 <= center[1] <= ref_y2:
                size = (x2 - x1) * (y2 - y1)
                size_diff = abs(size - ref_size) / ref_size
                if size_diff > self.threshold:
                    return True
            else:
                return False
        return False


class MovementDirectionDetector:
    def __init__(self, bboxes, threshold=3):
        """
        bboxes: list of (x1, y1, x2, y2)
        threshold: minimum pixel movement to count as real movement (default 3)
        """
        self.bboxes = bboxes
        self.threshold = threshold

    def _center(self, bbox):
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    def get_directions(self):
        """
        Returns a list of movement directions between consecutive frames,
        ignoring jitter smaller than the threshold.
        """
        directions = []
        prev_center = self._center(self.bboxes[0])
        for bbox in self.bboxes[1:]:
            curr_center = self._center(bbox)
            dx = curr_center[0] - prev_center[0]
            dy = curr_center[1] - prev_center[1]
            direction = "stationary"
            if abs(dx) > self.threshold or abs(dy) > self.threshold:
                if abs(dx) > abs(dy):
                    direction = "right" if dx > 0 else "left"
                else:
                    direction = "down" if dy > 0 else "up"
            directions.append(direction)
            prev_center = curr_center
        return directions

    def get_main_direction(self):
        """
        Returns the main direction of movement based on the most frequent direction,
        ignoring 'stationary'.
        """
        directions = self.get_directions()
        filtered = [d for d in directions if d != "stationary"]
        if not filtered:
            return "stationary"
        return Counter(filtered).most_common(1)[0][0]
