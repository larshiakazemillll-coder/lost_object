import cv2
import numpy as np
from typing import Union


def safe_draw_bboxes(frame: np.ndarray, bboxes: np.ndarray, color=(0, 255, 0)):
    if bboxes is None or bboxes.size == 0:
        return frame
    
    for row in bboxes:
        try:
            x1, y1, x2, y2 = map(int, row[:4])
            track_id = int(row[4]) if len(row) >= 5 else -1
            pt1 = (int(x1), int(y1))
            pt2 = (int(x2), int(y2))
            cv2.rectangle(frame, pt1, pt2, color, 2)
            cv2.putText(frame, f'ID:{track_id}', (pt1[0], max(pt1[1]-6, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        except Exception:
            continue
    return frame

# --- helpers for IoU / matching ---
def iou_xyxy(boxA, boxB):
    # boxes: [x1,y1,x2,y2]
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH
    areaA = max(0, boxA[2] - boxA[0]) * max(0, boxA[3] - boxA[1])
    areaB = max(0, boxB[2] - boxB[0]) * max(0, boxB[3] - boxB[1])
    union = areaA + areaB - interArea
    if union <= 0:
        return 0.0
    return interArea / union

def match_track_to_dets(track_box, det_boxes, iou_thresh=0.1):
    # return index of best matching det (or -1)
    best_iou = 0.0
    best_idx = -1
    for i, db in enumerate(det_boxes):
        iou = iou_xyxy(track_box, db[:4])
        if iou > best_iou:
            best_iou = iou
            best_idx = i
    return best_idx if best_iou >= iou_thresh else -1

def draw_bboxes_and_ids(frame, bboxes, cam_id):
    """Draw bounding boxes, camera ID, and track IDs on the frame.
    
    Args:
        frame: Input image frame
        bboxes: List of bounding boxes in format [x1, y1, x2, y2, track_id, ...]
        cam_id: Camera identifier
    
    Returns:
        Annotated frame with boxes and IDs
    """
    # Draw camera ID in top-left corner
    cv2.putText(frame, f"Cam: {cam_id}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Draw each bounding box and track ID
    # for bbox in bboxes:
    if len(bboxes) >= 5:  # Ensure bbox has track ID
        x1, y1, x2, y2, track_id = bboxes[:5]
        
        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Draw track ID above the bounding box
        cv2.putText(frame, f"ID: {track_id}", (x1, y1 - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    else:
        # Fallback if no track ID is available
        x1, y1, x2, y2 = bboxes[:4]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
    return frame

def draw_bboxes_and_ids(frame, bboxes, cam_id):
    for bbox in bboxes:
        x1, y1, x2, y2, obj_id, _ = bbox
        pt1 = (int(round(x1)), int(round(y1)))
        pt2 = (int(round(x2)), int(round(y2)))
        color = (0, 255, 0)
        thickness = 2
        cv2.rectangle(frame, pt1, pt2, color, thickness)
        label = f'ID {int(obj_id)}'
        cv2.putText(frame, label, (pt1[0], pt1[1] - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(frame, f'Cam {cam_id}', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    return frame

def resize_to_same_height(frames, h=400):
    out = []
    for frame in frames:
        aspect = frame.shape[1] / frame.shape[0]
        new_w = int(h * aspect)
        out.append(cv2.resize(frame, (new_w, h)))
    return out

def resize_to_same_height(frames):
    """Resize all frames to match the height of the smallest frame"""
    if not frames:
        return frames
    
    min_height = min(f.shape[0] for f in frames)
    resized_frames = []
    
    for frame in frames:
        if frame.shape[0] != min_height:
            scale = min_height / frame.shape[0]
            new_width = int(frame.shape[1] * scale)
            resized_frame = cv2.resize(frame, (new_width, min_height))
            resized_frames.append(resized_frame)
        else:
            resized_frames.append(frame)
    
    return resized_frames

def scale_boxes(img1_shape, boxes, img0_shape, ratio_pad=None, padding=True, xywh=False):
    """
    Rescale bounding boxes from img1_shape to img0_shape using NumPy.

    Args:
        img1_shape (tuple): The shape of the image that the bounding boxes are for, in the format of (height, width).
        boxes (numpy.ndarray): The bounding boxes of the objects in the image, in the format of (x1, y1, x2, y2).
        img0_shape (tuple): The shape of the target image, in the format of (height, width).
        ratio_pad (tuple): A tuple of (ratio, pad) for scaling the boxes. If not provided, the ratio and pad will be
            calculated based on the size difference between the two images.
        padding (bool): If True, assuming the boxes is based on image augmented by yolo style. If False then do regular
            rescaling.
        xywh (bool): The box format is xywh or not.

    Returns:
        (numpy.ndarray): The scaled bounding boxes, in the format of (x1, y1, x2, y2).
    """
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (
            round((img1_shape[1] - img0_shape[1] * gain) / 2 - 0.1),
            round((img1_shape[0] - img0_shape[0] * gain) / 2 - 0.1),
        )  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    boxes = boxes.copy()  # Create a copy to avoid modifying the original array
    if padding:
        boxes[..., 0] -= pad[0]  # x padding
        boxes[..., 1] -= pad[1]  # y padding
        if not xywh:
            boxes[..., 2] -= pad[0]  # x padding
            boxes[..., 3] -= pad[1]  # y padding
    boxes[..., :4] /= gain
    return clip_boxes(boxes, img0_shape)

def clip_boxes(boxes, shape):
    """
    Clip bounding boxes to image shape (height, width) using NumPy.

    Args:
        boxes (numpy.ndarray): Bounding boxes to clip, in (x1, y1, x2, y2) format.
        shape (tuple): Image shape (height, width).

    Returns:
        (numpy.ndarray): Clipped bounding boxes.
    """
    boxes[..., [0, 2]] = boxes[..., [0, 2]].clip(0, shape[1])  # x1, x2
    boxes[..., [1, 3]] = boxes[..., [1, 3]].clip(0, shape[0])  # y1, y2
    return boxes

def xywh2xyxy(x: np.ndarray) -> np.ndarray:
    """
    Convert bounding boxes from (x_center, y_center, w, h)
    to (x1, y1, x2, y2).
    """
    assert x.shape[-1] == 4, f"Expected last dim=4, got {x.shape[-1]}"
    y = empty_like(x)
    xy = x[..., :2]
    wh = x[..., 2:] * 0.5
    y[..., :2] = xy - wh  # top-left
    y[..., 2:] = xy + wh  # bottom-right
    return y

def empty_like(x: np.ndarray) -> np.ndarray:
    """
    Create an uninitialized NumPy array of the same shape as x, dtype float32.
    """
    return np.empty_like(x, dtype=np.float32)


class LetterBox:
    """
    Resize image and padding for detection, instance segmentation, pose.

    This class resizes and pads images to a specified shape while preserving aspect ratio. It also updates
    corresponding labels and bounding boxes.

    Attributes:
        new_shape (tuple): Target shape (height, width) for resizing.
        auto (bool): Whether to use minimum rectangle.
        scale_fill (bool): Whether to stretch the image to new_shape.
        scaleup (bool): Whether to allow scaling up. If False, only scale down.
        stride (int): Stride for rounding padding.
        center (bool): Whether to center the image or align to top-left.

    Methods:
        __call__: Resize and pad image, update labels and bounding boxes.

    Examples:
        >>> transform = LetterBox(new_shape=(640, 640))
        >>> result = transform(labels)
        >>> resized_img = result["img"]
        >>> updated_instances = result["instances"]
    """

    def __init__(self, new_shape=(640, 640), auto=False, scale_fill=False, scaleup=True, center=True, stride=32):
        """
        Initialize LetterBox object for resizing and padding images.

        This class is designed to resize and pad images for object detection, instance segmentation, and pose estimation
        tasks. It supports various resizing modes including auto-sizing, scale-fill, and letterboxing.

        Args:
            new_shape (Tuple[int, int]): Target size (height, width) for the resized image.
            auto (bool): If True, use minimum rectangle to resize. If False, use new_shape directly.
            scale_fill (bool): If True, stretch the image to new_shape without padding.
            scaleup (bool): If True, allow scaling up. If False, only scale down.
            center (bool): If True, center the placed image. If False, place image in top-left corner.
            stride (int): Stride of the model (e.g., 32 for YOLOv5).

        Attributes:
            new_shape (Tuple[int, int]): Target size for the resized image.
            auto (bool): Flag for using minimum rectangle resizing.
            scale_fill (bool): Flag for stretching image without padding.
            scaleup (bool): Flag for allowing upscaling.
            stride (int): Stride value for ensuring image size is divisible by stride.

        Examples:
            >>> letterbox = LetterBox(new_shape=(640, 640), auto=False, scale_fill=False, scaleup=True, stride=32)
            >>> resized_img = letterbox(original_img)
        """
        self.new_shape = new_shape
        self.auto = auto
        self.scale_fill = scale_fill
        self.scaleup = scaleup
        self.stride = stride
        self.center = center  # Put the image in the middle or top-left

    def __call__(self, labels=None, image=None):
        """
        Resizes and pads an image for object detection, instance segmentation, or pose estimation tasks.

        This method applies letterboxing to the input image, which involves resizing the image while maintaining its
        aspect ratio and adding padding to fit the new shape. It also updates any associated labels accordingly.

        Args:
            labels (Dict | None): A dictionary containing image data and associated labels, or empty dict if None.
            image (np.ndarray | None): The input image as a numpy array. If None, the image is taken from 'labels'.

        Returns:
            (Dict | Tuple): If 'labels' is provided, returns an updated dictionary with the resized and padded image,
                updated labels, and additional metadata. If 'labels' is empty, returns a tuple containing the resized
                and padded image, and a tuple of (ratio, (left_pad, top_pad)).

        Examples:
            >>> letterbox = LetterBox(new_shape=(640, 640))
            >>> result = letterbox(labels={"img": np.zeros((480, 640, 3)), "instances": Instances(...)})
            >>> resized_img = result["img"]
            >>> updated_instances = result["instances"]
        """
        if labels is None:
            labels = {}
        img = labels.get("img") if image is None else image
        shape = img.shape[:2]  # current shape [height, width]
        new_shape = labels.pop("rect_shape", self.new_shape)
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not self.scaleup:  # only scale down, do not scale up (for better val mAP)
            r = min(r, 1.0)

        # Compute padding
        ratio = r, r  # width, height ratios
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
        if self.auto:  # minimum rectangle
            dw, dh = np.mod(dw, self.stride), np.mod(dh, self.stride)  # wh padding
        elif self.scale_fill:  # stretch
            dw, dh = 0.0, 0.0
            new_unpad = (new_shape[1], new_shape[0])
            ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

        if self.center:
            dw /= 2  # divide padding into 2 sides
            dh /= 2

        if shape[::-1] != new_unpad:  # resize
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
            if img.ndim == 2:
                img = img[..., None]

        top, bottom = int(round(dh - 0.1)) if self.center else 0, int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)) if self.center else 0, int(round(dw + 0.1))
        h, w, c = img.shape
        if c == 3:
            img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
        else:  # multispectral
            pad_img = np.full((h + top + bottom, w + left + right, c), fill_value=114, dtype=img.dtype)
            pad_img[top : top + h, left : left + w] = img
            img = pad_img

        if labels.get("ratio_pad"):
            labels["ratio_pad"] = (labels["ratio_pad"], (left, top))  # for evaluation

        if len(labels):
            labels = self._update_labels(labels, ratio, left, top)
            labels["img"] = img
            labels["resized_shape"] = new_shape
            return labels
        else:
            return img

def non_max_suppression(
    prediction: np.ndarray,
    conf_thres: float = 0.25,
    iou_thres: float = 0.45,
    classes: int | None = None,
    agnostic: bool = False,
    max_det: int = 300,
    nc: int = 0,  # number of classes
) -> list[np.ndarray]:
    """
    Pure-Numpy NMS.

    Args:
        prediction: array of shape (B, nc+4, N) or (B, N, nc+4).
        conf_thres: filter boxes with confidence <= this.
        iou_thres: IoU threshold for suppressing overlaps.
        classes: if int, only keep that class; if None, keep all.
        agnostic: if False and nc>1, do class-aware NMS.
        max_det: max boxes per image.
        nc: number of classes in prediction.

    Returns:
        List of length B of arrays (M,6): [x1,y1,x2,y2,conf,cls].
    """
    # unify shape to (B, N, 4+nc)
    if prediction.ndim == 3 and prediction.shape[1] == nc + 4:
        prediction = np.transpose(prediction, (0, 2, 1))
    B, N, _ = prediction.shape
    outputs: list[np.ndarray] = []

    def compute_iou(box, boxes):
        # box: (4,), boxes: (M,4)
        xx1 = np.maximum(box[0], boxes[:,0])
        yy1 = np.maximum(box[1], boxes[:,1])
        xx2 = np.minimum(box[2], boxes[:,2])
        yy2 = np.minimum(box[3], boxes[:,3])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        area1 = (box[2]-box[0])*(box[3]-box[1])
        area2 = (boxes[:,2]-boxes[:,0])*(boxes[:,3]-boxes[:,1])
        return inter / (area1 + area2 - inter + 1e-16)

    for img in prediction:
        # 1) extract boxes, confidences, classes
        boxes = xywh2xyxy(img[:, :4])
        if nc > 1:
            cls_scores = img[:, 4:4+nc]
            cls_idx = np.argmax(cls_scores, axis=1)
            conf = cls_scores[np.arange(N), cls_idx]
            cls  = cls_idx.astype(np.float32)
        else:
            conf = img[:, 4].astype(np.float32)
            cls  = np.zeros_like(conf)

        # 2) confidence filter
        mask = conf > conf_thres
        boxes, conf, cls = boxes[mask], conf[mask], cls[mask]
        if classes is not None:
            class_mask = cls == float(classes)
            boxes, conf, cls = boxes[class_mask], conf[class_mask], cls[class_mask]

        # 3) if empty, append empty (0×6) array
        if boxes.shape[0] == 0:
            outputs.append(np.zeros((0,6), dtype=np.float32))
            continue

        # 4) prepare for NMS
        order = conf.argsort()[::-1]
        keep = []
        while order.size and len(keep) < max_det:
            i = order[0]
            keep.append(i)
            if order.size == 1:
                break
            ious = compute_iou(boxes[i], boxes[order[1:]])
            # determine which to keep next
            if nc > 1 and not agnostic:
                # suppress only same-class boxes
                same_cls = cls[order[1:]] == cls[i]
                suppress = (ious > iou_thres) & same_cls
            else:
                suppress = ious > iou_thres
            order = order[1:][~suppress]

        keep = np.array(keep, dtype=np.int32)
        # 5) pack detections
        det = np.concatenate([
            boxes[keep],
            conf[keep, None],
            cls[keep, None]
        ], axis=1)
        outputs.append(det.astype(np.float32))

    return outputs

def process_track_results(results, thresh=0.30):
    """
    Processes tracking results by filtering detections based on confidence and class index.

    Parameters
    ----------
    results : list
        The list of detection results in the format [x1, y1, x2, y2, confidence, cls_index].
    thresh : float, optional
        The minimum confidence threshold for filtering detections. Defaults to 0.55.

    Returns
    -------
    boxes : np.ndarray
        The filtered bounding boxes in the format [center_x, center_y, width, height].
    scores : np.ndarray
        The confidence scores of the filtered detections.
    labels : np.ndarray
        The class labels of the filtered detections.

    Raises
    ------
    TypeError
        If the input results are not a list or if the threshold is not a float.
    ValueError
        If the input results are empty or if the threshold is invalid.
    """
    result = []
    # Define the class index for person
    PERSON_CLASS_INDEX = 0
    # Process each detection result
    for (*xyxy, confidence, cls_index) in results:
        cls_index = int(cls_index.item())
        confidence = float(confidence.item())
        # Filter by class index and confidence threshold
        if cls_index == PERSON_CLASS_INDEX and confidence >= thresh:
            x1, y1, x2, y2 =  list(map(int, xyxy))
            result.append([x1, y1, x2, y2, confidence, cls_index])

    if len(result) == 0:
        result = np.array([])

    return np.array(result)

def process_detect_results(results, thresh=0.55):
    """
    Processes detection results by filtering boxes based on confidence and class index.

    Parameters
    ----------
    results : list
        The list of detection results in the format [x1, y1, x2, y2, confidence, cls_index].
    thresh : float, optional
        The minimum confidence threshold for filtering detections. Defaults to 0.55.

    Returns
    -------
    boxes : np.ndarray
        The filtered bounding boxes in the format [x1, y1, x2, y2].

    Raises
    ------
    TypeError
        If the input results are not a list or if the threshold is not a float.
    ValueError
        If the input results are empty or if the threshold is invalid.
    """
    # Initialize list for boxes 
    boxes = []
    # Define the class index for person
    PERSON_CLASS_INDEX = 0
    # Process each detection result
    for (*xyxy, confidence, cls_index) in results:

        cls_index = int(cls_index.item())
        confidence = float(confidence.item())

        # Filter by class index and confidence threshold
        if cls_index == PERSON_CLASS_INDEX and confidence >= thresh:
            x1, y1, x2, y2 = [int(x.item()) for x in xyxy]
            boxes.append([x1, y1, x2, y2])
    # Handle empty results
    if len(boxes) == 0: 
        boxes = np.array([]).reshape(0, 4)
    return np.array(boxes)

def preprocess_image(
    image_path: Union[str, np.ndarray],
    imgsz: int = 640,
    fp16: bool = False,
    return_original: bool = False
) -> Union[np.ndarray, tuple]:
    """
    Preprocess image without PyTorch (returns NumPy array instead).

    Args:
        image_path: Path to image or numpy array (BGR/HWC format)
        imgsz: Target size (default: 640)
        fp16: Use float16 instead of float32 (default: False)
        return_original: If True, returns (processed, original_image)
    
    Returns:
        np.ndarray: Shape [1, 3, imgsz, imgsz] (normalized 0-1)
        OR tuple: (processed, original_image) if return_original=True
    """
    # 1. Load image (if path provided)
    if isinstance(image_path, str):
        img = cv2.imread(image_path)  # BGR, HWC, uint8 (0-255)
        if img is None:
            raise FileNotFoundError(f"Image not found at {image_path}")
    else:
        img = image_path  # Assume numpy array input

    original_img = img.copy()

    # 2. LetterBox transform (resize + pad)
    letterbox = LetterBox(new_shape=imgsz, auto=False, stride=32)
    img = letterbox(image=img)  # Padded to imgsz x imgsz

    # 3. Convert BGR->RGB, HWC->CHW
    img = img[..., ::-1].transpose(2, 0, 1)  # BGR to RGB, HWC to CHW
    img = np.ascontiguousarray(img)  # Ensure memory continuity

    # 4. Normalize to [0, 1] and convert dtype
    img = img.astype(np.float16 if fp16 else np.float32)  # No PyTorch!
    img /= 255.0  # Normalize

    # 5. Add batch dimension [C,H,W] -> [1,C,H,W] (NumPy expand_dims)
    img = np.expand_dims(img, axis=0)

    return (img, original_img) if return_original else img

def postprocess(pred_boxes, input_hw, orig_img, min_conf_threshold=0.25, nms_iou_threshold=0.1, nc=80, classess=0):
    """
    YOLOv8 model postprocessing function. Applied non maximum supression algorithm to detections and rescale boxes to original image size
    Parameters:
        pred_boxes (np.ndarray): model output prediction boxes
        input_hw (np.ndarray): preprocessed image
        orig_image (np.ndarray): image before preprocessing
        min_conf_threshold (float, *optional*, 0.25): minimal accepted confidence for object filtering
        nms_iou_threshold (float, *optional*, 0.45): minimal overlap score for removing objects duplicates in NMS
        agnostic_nms (bool, *optiona*, False): apply class agnostinc NMS approach or not
        max_detections (int, *optional*, 300):  maximum detections after NMS
    Returns:
       pred (List[Dict[str, np.ndarray]]): list of dictionary with det - detected boxes in format [x1, y1, x2, y2, score, label]
    """



    # Non-maximum suppression parameters    
    nms_kwargs = {"agnostic": False, "max_det": 100}
    
    
    preds = non_max_suppression(
        pred_boxes[:, :, :],
        min_conf_threshold,
        nms_iou_threshold,
        nc=nc,
        classes = classess,
        **nms_kwargs,
    )
    
    # Initialize results list
    results = []
    # Process each prediction

    for i, pred in enumerate(preds):
        # Get the original image shape
        shape = orig_img[i].shape if isinstance(orig_img, list) else orig_img.shape
        # Handle empty predictions
        if not len(pred):
            results.append({"det": np.array([]), "segment": np.array([])})
            continue
        pred[:, :4] = scale_boxes(input_hw, pred[:, :4], shape).round()
        
        pred_np = pred
        
        # Append the processed prediction to results
        results.append({"det": pred_np})
    return results
