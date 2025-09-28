import argparse
import json
import logging as log
import os
import sys
import time
from collections import defaultdict, deque
from pathlib import Path
from typing import Tuple, List, Dict

import cv2
import numpy as np
import supervision as sv
import torch
from openvino import runtime as ov
from supervision import ColorLookup
from ultralytics import YOLO
from ultralytics.utils import ops

from deep_sort_realtime.deepsort_tracker import DeepSort

CATEGORIES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant",
    "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
    "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
    "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet",
    "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
    "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
]

# --------- Utility Functions (replacing demo_utils) ---------

def available_devices():
    """Return a mapping of available OpenVINO devices."""
    core = ov.Core()
    devices = core.available_devices
    return {dev: dev for dev in devices}

def draw_text(frame, text, point=(10, 30), color=(0, 255, 0), scale=0.8, thickness=2, bg_color=(0,0,0)):
    """Draw text with background for readability."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_size, _ = cv2.getTextSize(text, font, scale, thickness)
    x, y = point
    cv2.rectangle(frame, (x, y - text_size[1] - 4), (x + text_size[0] + 4, y + 4), bg_color, -1)
    cv2.putText(frame, text, (x + 2, y), font, scale, color, thickness, cv2.LINE_AA)

def draw_control_panel(frame, device_mapping):
    """Draws a simple control panel with device info."""
    y = 80
    draw_text(frame, "Device Switch: [1-9] | Model: [f] FP16, [i] INT8", point=(10, y), color=(255,255,0))
    y += 30
    draw_text(frame, "Available Devices: " + ", ".join(device_mapping.keys()), point=(10, y), color=(255,255,0))

def draw_ov_watermark(frame):
    """Draws a watermark for OpenVINO."""
    h, w = frame.shape[:2]
    text = "OpenVINO"
    draw_text(frame, text, point=(w-180, h-20), color=(255,255,255), bg_color=(0,0,0))

# -----------------------------------------------------------

def convert(model_name: str, model_dir: Path) -> tuple[Path, Path]:
    model_path = model_dir / f"{model_name}.pt"
    yolo_model = YOLO(model_path)
    ov_model_path = model_dir / f"{model_name}_openvino_model"
    ov_int8_model_path = model_dir / f"{model_name}_int8_openvino_model"
    if not ov_model_path.exists():
        ov_model_path = yolo_model.export(format="openvino", dynamic=False, half=True)
    if not ov_int8_model_path.exists():
        ov_int8_model_path = yolo_model.export(format="openvino", dynamic=False, half=True, int8=True, data="coco128.yaml")
    return Path(ov_model_path) / f"{model_name}.xml", Path(ov_int8_model_path) / f"{model_name}.xml"

def letterbox(img: np.ndarray, new_shape: Tuple[int, int]) -> Tuple[np.ndarray, Tuple[float, float], Tuple[int, int]]:
    shape = img.shape[1::-1]
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    ratio = r, r
    new_unpad = int(round(shape[0] * r)), int(round(shape[1] * r))
    dw, dh = new_shape[0] - new_unpad[0], new_shape[1] - new_unpad[1]
    dw /= 2
    dh /= 2
    if shape != new_unpad:
        img = cv2.resize(img, dsize=new_unpad, interpolation=cv2.INTER_AREA)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(128, 128, 128))
    return img, ratio, (int(dw), int(dh))

def preprocess(image: np.ndarray, input_size: Tuple[int, int]) -> Tuple[np.ndarray, Tuple[int, int]]:
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    l = clahe.apply(l)
    image = cv2.merge([l, a, b])
    image = cv2.cvtColor(image, cv2.COLOR_LAB2BGR)
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    image = cv2.filter2D(image, -1, kernel)
    image, _, padding = letterbox(image, new_shape=input_size)
    image = image.astype(np.float32)
    image /= 255.0
    image = image.transpose((2, 0, 1))
    image = np.expand_dims(image, axis=0)
    return image, padding

def postprocess(pred_boxes: np.ndarray, pred_masks: np.ndarray, input_size: Tuple[int, int], orig_img: np.ndarray, padding: Tuple[int, int], category_id: int,
                min_conf_threshold: float = 0.15, nms_iou_threshold: float = 0.45, agnostic_nms: bool = False, max_detections: int = 200) -> sv.Detections:
    nms_kwargs = {"agnostic": agnostic_nms, "max_det": max_detections}
    pred = ops.non_max_suppression(torch.from_numpy(pred_boxes), min_conf_threshold, nms_iou_threshold, nc=80, **nms_kwargs)[0]
    if not len(pred):
        return sv.Detections.empty()
    masks = pred_masks
    if pred_masks is not None:
        masks = np.array(ops.process_mask(torch.from_numpy(pred_masks[0]), pred[:, 6:], pred[:, :4], input_size, upsample=True))
        masks = np.array([cv2.resize(mask[padding[1]:-padding[1] - 1, padding[0]:-padding[0] - 1], orig_img.shape[:2][::-1], interpolation=cv2.INTER_AREA) for mask in masks])
        masks = masks.astype(np.bool_)
    pred[:, :4] = ops.scale_boxes(input_size, pred[:, :4], orig_img.shape).round()
    pred = np.array(pred)
    xyxy = pred[:, :4].astype(np.float32)
    confidence = pred[:, 4].astype(np.float32)
    class_id = pred[:, 5].astype(np.int32)
    global det
    det = sv.Detections(xyxy=xyxy, mask=masks, confidence=confidence, class_id=class_id)
    filtered_det = det[det.class_id == category_id]
    return filtered_det

def get_model(model_path: Path, device: str = "AUTO") -> ov.CompiledModel:
    core = ov.Core()
    model = core.read_model(model_path)
    model = core.compile_model(model, device_name=device, config={"PERFORMANCE_HINT": "LATENCY", "CACHE_DIR": "cache"})
    return model

def load_zones(json_path: str) -> List[np.ndarray]:
    with open(json_path) as f:
        zones_dict = json.load(f)
    return [np.array(zone["points"], np.int32) for zone in zones_dict.values()]

def get_annotators(json_path: str, resolution_wh: Tuple[int, int], colorful: bool = False) -> Tuple[List, List, List, List, List]:
    polygons = load_zones(json_path)
    colors = sv.ColorPalette.DEFAULT
    zones = []
    zone_annotators = []
    box_annotators = []
    masks_annotators = []
    label_annotators = []
    for index, polygon in enumerate(polygons, start=1):
        zone = sv.PolygonZone(polygon=polygon, frame_resolution_wh=resolution_wh)
        zones.append(zone)
        zone_annotators.append(sv.PolygonZoneAnnotator(zone=zone, color=colors.by_idx(index), thickness=0))
        box_annotator = sv.BoxAnnotator(color_lookup=ColorLookup.INDEX) if colorful else sv.BoxAnnotator(color=colors.by_idx(index))
        box_annotators.append(box_annotator)
        mask_annotator = sv.MaskAnnotator(color_lookup=ColorLookup.INDEX) if colorful else sv.MaskAnnotator(color=colors.by_idx(index))
        masks_annotators.append(mask_annotator)
        label_annotator = sv.LabelAnnotator(text_scale=0.7, color_lookup=ColorLookup.INDEX) if colorful else sv.LabelAnnotator(text_scale=0.7, color=colors.by_idx(index))
        label_annotators.append(label_annotator)
    return zones, zone_annotators, box_annotators, masks_annotators, label_annotators

def track_objects(frame: np.array, detections: sv.Detections, tracker: DeepSort) -> List:
    detection_list = []
    for det in zip(detections.xyxy, detections.confidence):
        bbox = det[0].tolist()
        confidence = det[1]
        detection_list.append((bbox, confidence))
    tracks = tracker.update_tracks(detection_list, frame=frame)
    return list(sorted(tracks, key=lambda x: x.det_conf if x.det_conf is not None else 0.0, reverse=True))

# --- Velocity tracking state ---
# For each track_id, store last center position
track_last_centers = {}
track_last_frame = {}

def compute_net_velocity(tracks, frame_idx):
    """
    Compute net velocity (average dx, dy per frame) for all tracks with valid movement.
    Returns: (avg_dx, avg_dy, direction_str)
    """
    global track_last_centers, track_last_frame
    total_dx = 0.0
    total_dy = 0.0
    count = 0
    for track in tracks:
        if not hasattr(track, "track_id") or track.track_id is None:
            continue
        if track.bbox is None:
            continue
        # Get current center
        x1, y1, x2, y2 = track.bbox
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        tid = track.track_id
        if tid in track_last_centers and tid in track_last_frame:
            last_cx, last_cy = track_last_centers[tid]
            last_f = track_last_frame[tid]
            dt = frame_idx - last_f
            if dt > 0:
                dx = (cx - last_cx) / dt
                dy = (cy - last_cy) / dt
                total_dx += dx
                total_dy += dy
                count += 1
        # Update last center and frame
        track_last_centers[tid] = (cx, cy)
        track_last_frame[tid] = frame_idx
    if count == 0:
        return 0.0, 0.0, "none"
    avg_dx = total_dx / count
    avg_dy = total_dy / count
    # Determine direction
    dir_x = ""
    dir_y = ""
    if abs(avg_dx) > 1:
        dir_x = "right" if avg_dx > 0 else "left"
    if abs(avg_dy) > 1:
        dir_y = "down" if avg_dy > 0 else "up"
    if dir_x and dir_y:
        direction = f"{dir_y}-{dir_x}"
    elif dir_x:
        direction = dir_x
    elif dir_y:
        direction = dir_y
    else:
        direction = "static"
    return avg_dx, avg_dy, direction

def draw_annotations(frame: np.array, detections: sv.Detections, tracker: DeepSort, queue_count: Dict, object_limit: int, category:str,
                     zones: List, zone_annotators: List, box_annotators: List, masks_annotators: List, label_annotators: List, frame_idx: int = 0) -> None:

    for zone_annotator in zone_annotators:
        frame = zone_annotator.annotate(scene=frame)

    tracks = []
    if detections:
        tracks = track_objects(frame, detections, tracker)
        for zone_id, (zone, box_annotator, masks_annotator, label_annotator) in enumerate(
                zip(zones, box_annotators, masks_annotators, label_annotators), start=1):
            mask = zone.trigger(detections=detections)
            detections_filtered = detections[mask]
            frame = masks_annotator.annotate(scene=frame, detections=detections_filtered)
            frame = box_annotator.annotate(scene=frame, detections=detections_filtered)
            det_count = len(detections_filtered)
            label_annotator.annotate(scene=frame, detections=detections_filtered,
                                     labels=[f"ID: {track.track_id if track.time_since_update == 0 else ' '}" for
                                             track, in_zone in zip(tracks, mask) if in_zone])
            queue_count[zone_id].append(det_count)
            room_capacity = 20
            occupancy_percentage = (det_count / room_capacity) * 100 if room_capacity > 0 else 0
            safety_status = "SAFE" if occupancy_percentage <= 100 else "UNSAFE"
            stampede_risk = 0
            if occupancy_percentage > 100:
                excess_percentage = occupancy_percentage - 100
                if excess_percentage <= 20:
                    stampede_risk = min(15 + (excess_percentage * 2), 30)
                elif excess_percentage <= 50:
                    stampede_risk = min(30 + ((excess_percentage - 20) * 1.5), 60)
                else:
                    stampede_risk = min(60 + ((excess_percentage - 50) * 0.8), 85)
            # Compute net velocity for all tracks
            avg_dx, avg_dy, direction = compute_net_velocity(tracks, frame_idx)
            # Print log with velocity
            log.info(f"{det_count}/{room_capacity} people "
                     f"({occupancy_percentage:.1f}% occupied) - Status: {safety_status} | "
                     f"Net velocity: ({avg_dx:.2f}, {avg_dy:.2f}) px/frame [{direction}]")
            if occupancy_percentage > 100:
                log.warning(f"OVERCAPACITY! Stampede risk: {stampede_risk:.1f}%")
            safety_text = f"{det_count}/{room_capacity} people - {occupancy_percentage:.1f}% occupied ({safety_status})"
            if occupancy_percentage > 100:
                safety_text += f" | Stampede Risk: {stampede_risk:.1f}%"
            # Add velocity to overlay
            velocity_text = f"Net velocity: ({avg_dx:.2f}, {avg_dy:.2f}) px/frame [{direction}]"
            draw_text(frame, text=safety_text, point=(10, 30 + zone_id * 30))
            draw_text(frame, text=velocity_text, point=(10, 30 + zone_id * 30 + 22), color=(0, 200, 255))

def run(
    video_path: str,
    model_paths: Tuple[Path, Path],
    model_name: str = "",
    category: str = "person",
    zones_config_file: str = "",
    object_limit: int = 3,
    room_capacity: int = 50,
    tracker_frames: int = 1800,
    flip: bool = True,
    colorful: bool = False,
    last_frames: int = 50,
    output_path: str = "output.mp4"
) -> None:
    log.getLogger().setLevel(log.INFO)
    model_mapping = {
        "FP16": model_paths[0],
        "INT8": model_paths[1],
    }
    device_mapping = available_devices()
    model_type = "INT8"
    device_type = "AUTO"
    core = ov.Core()
    core.set_property({"CACHE_DIR": "cache"})
    model = get_model(model_mapping[model_type], device_type)
    input_shape = tuple(model.inputs[0].shape)[:0:-1]

    # Open input video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Failed to open video: {video_path}")
        return

    # Get video properties
    f_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    f_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Prepare output video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (f_width, f_height))

    # Prepare annotators and tracker
    zones, zone_annotators, box_annotators, masks_annotators, label_annotators = get_annotators(
        json_path=zones_config_file, resolution_wh=(f_width, f_height), colorful=colorful
    )
    category_id = CATEGORIES.index(category)
    queue_count = defaultdict(lambda: deque(maxlen=last_frames))
    processing_times = deque(maxlen=100)
    tracker = DeepSort(max_age=tracker_frames, n_init=3)

    frame_idx = 0
    window_name = "Stampede Prevention - Live"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, f_width, f_height)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Source ended")
            break
        if flip:
            frame = cv2.flip(frame, 1)
        frame = np.array(frame)
        input_image, padding = preprocess(image=frame, input_size=input_shape[:2])
        start_time = time.time()
        results = model(input_image)
        processing_times.append(time.time() - start_time)
        boxes = results[model.outputs[0]]
        masks = results[model.outputs[1]] if len(model.outputs) > 1 else None
        detections = postprocess(
            pred_boxes=boxes,
            pred_masks=masks,
            input_size=input_shape[:2],
            orig_img=frame,
            padding=padding,
            category_id=category_id,
            min_conf_threshold=0.15,
            nms_iou_threshold=0.45
        )
        draw_annotations(
            frame, detections, tracker, queue_count, object_limit, category,
            zones, zone_annotators, box_annotators, masks_annotators, label_annotators, frame_idx=frame_idx
        )
        processing_time = np.mean(processing_times) * 1000
        fps_calc = 1000 / processing_time if processing_time > 0 else 0
        draw_text(frame, text=f"Inference time: {processing_time:.0f}ms ({fps_calc:.1f} FPS)", point=(f_width * 3 // 5, 10))
        draw_text(frame, text=f"Currently running {model_name} ({model_type}) on {device_type}", point=(f_width * 3 // 5, 50))
        draw_control_panel(frame, device_mapping)
        draw_ov_watermark(frame)
        out.write(frame)

        # Show the video live
        cv2.imshow(window_name, frame)
        key = cv2.waitKey(1) & 0xFF
        # Press 'q' to quit early
        if key == ord('q'):
            print("Quitting live display.")
            break

        frame_idx += 1
        if frame_idx % 50 == 0 or frame_idx == 1:
            print(f"Processed {frame_idx}/{total_frames} frames...")

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Annotated video saved to {output_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--stream', default="demo1.mp4", type=str, help="Path to a video file")
    parser.add_argument("--model_name", type=str, default="yolo11m", help="Model version to be converted",
                        choices=["yolov8n", "yolov8s", "yolov8m", "yolov8l", "yolov8x", "yolov8n-seg", "yolov8s-seg", "yolov8m-seg", "yolov8l-seg", "yolov8x-seg",
                                 "yolo11n", "yolo11s", "yolo11m", "yolo11l", "yolo11x", "yolo11n-seg", "yolo11s-seg", "yolo11m-seg", "yolo11l-seg", "yolo11x-seg"])
    parser.add_argument("--model_dir", type=str, default="model", help="Directory to place the model in")
    parser.add_argument('--category', type=str, default="person", choices=CATEGORIES, help="The category to detect (from COCO dataset)")
    parser.add_argument('--zones_config_file', type=str, default="zones.json", help="Path to the zone config file (json)")
    parser.add_argument('--object_limit', type=int, default=3, help="The maximum number of objects in the area")
    parser.add_argument("--flip", type=bool, default=True, help="Mirror input video")
    parser.add_argument('--colorful', action="store_true", help="If objects should be annotated with random colors")
    parser.add_argument('--tracker_frames', type=int, default=1800, help="Maximum number of missed frames for the tracker")
    parser.add_argument('--room_capacity', type=int, default=50, help="Maximum safe capacity of the room")
    parser.add_argument('--output', type=str, default="output.mp4", help="Path to save the annotated output video")
    args = parser.parse_args()
    model_paths = convert(args.model_name, Path(args.model_dir))
    run(
        args.stream,
        model_paths,
        args.model_name,
        args.category,
        args.zones_config_file,
        args.object_limit,
        args.room_capacity,
        args.tracker_frames,
        args.flip,
        args.colorful,
        output_path=args.output
    )
