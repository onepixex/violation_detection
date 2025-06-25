import os
import cv2
import numpy as np
import logging
import time
import subprocess
from collections import defaultdict
from django.conf import settings
from ultralytics import YOLO

# Настройка логгирования
logging.getLogger("ultralytics").setLevel(logging.WARNING)

# Загрузка моделей
detection_model = YOLO(str(settings.YOLO_MODELS_DIR / 'detect_road_object.pt'))
segmentation_model = YOLO(str(settings.YOLO_MODELS_DIR / 'solid_line_seg.pt'))

# Шаблон нарушений
violations_template = {
    'Опасное сближение автомобилей': 0,
    'Непропуск пешехода': 0,
    'Проезд на красный': 0,
    'Пересечение сплошной линии': 0
}

# Словари для отслеживания времени нарушения
violation_start_times = defaultdict(lambda: None)
violation_cooldowns = defaultdict(lambda: 0)
violation_labels = {
    'car_approach': 'Опасное сближение автомобилей',
    'pedestrian': 'Непропуск пешехода',
    'red_light': 'Проезд на красный',
    'line_crossing': 'Пересечение сплошной линии'
}


def calculate_distance(box1, box2):
    x1, y1, x2, y2 = box1
    x3, y3, x4, y4 = box2
    center1 = ((x1 + x2) / 2, (y1 + y2) / 2)
    center2 = ((x3 + x4) / 2, (y3 + y4) / 2)
    return np.hypot(center1[0] - center2[0], center1[1] - center2[1])


def get_solid_line_mask(frame):
    seg_results = segmentation_model(frame, verbose=False)
    masks = seg_results[0].masks
    if masks is not None:
        mask_array = masks.data.cpu().numpy()
        binary_mask = np.any(mask_array > 0.5, axis=0).astype(np.uint8) * 255
    else:
        binary_mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
    return binary_mask


def check_violation(key, condition, current_time, duration_threshold=2.0, cooldown=5.0):
    if condition:
        if violation_start_times[key] is None:
            violation_start_times[key] = current_time
        elif (current_time - violation_start_times[key]) >= duration_threshold:
            if (current_time - violation_cooldowns[key]) >= cooldown:
                violation_cooldowns[key] = current_time
                return True
    else:
        violation_start_times[key] = None
    return False


def convert_to_h264(input_path, output_path=None):
    """Конвертирует видео в формат H.264"""
    if output_path is None:
        base, ext = os.path.splitext(input_path)
        output_path = f"{base}_h264{ext}"

    command = [
        'ffmpeg',
        '-y',
        '-i', input_path,
        '-c:v', 'libx264',
        '-preset', 'slow',
        '-crf', '28',
        '-pix_fmt', 'yuv420p',
        '-c:a', 'aac',
        '-b:a', '128k',
        '-movflags', '+faststart',
        output_path
    ]
    subprocess.run(command, check=True)
    return output_path


def process_video(video_path, output_folder, progress_callback=None):
    violations = violations_template.copy()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Не удалось открыть видео {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    os.makedirs(output_folder, exist_ok=True)

    temp_output_path = os.path.join(output_folder, f"temp_{os.path.basename(video_path)}")
    final_output_path = os.path.splitext(temp_output_path)[0] + ".mp4"

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0 or np.isnan(fps):
        fps = 25

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(temp_output_path, fourcc, fps, (width, height))

    start_time = time.time()
    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        current_time = time.time() - start_time
        solid_mask = get_solid_line_mask(frame)

        results = detection_model(frame, verbose=False)
        current_objects = {'Car': [], 'Red traffic light': [], 'pedestrian': []}

        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy()

            for box, cls in zip(boxes, classes):
                label = detection_model.names[int(cls)]
                x1, y1, x2, y2 = map(int, box)
                if label in current_objects:
                    current_objects[label].append((x1, y1, x2, y2))

        _check_car_approach(current_objects, violations, current_time, frame)
        _check_pedestrian_violation(current_objects, violations, current_time, frame)
        _check_red_light_violation(current_objects, violations, current_time, frame)
        _check_line_crossing(current_objects, solid_mask, violations, current_time, frame)

        _draw_objects(frame, current_objects)
        out.write(frame)

        frame_idx += 1
        if progress_callback and frame_idx % 5 == 0:
            progress_callback(int(frame_idx / total_frames * 100))

    cap.release()
    out.release()

    # Конвертация в H.264 mp4
    final_output_path = temp_output_path.replace(".mp4", "_h264.mp4")
    convert_to_h264(temp_output_path, final_output_path)
    os.remove(temp_output_path)

    return final_output_path, violations


def _check_car_approach(objects, violations, current_time, frame):
    for i, car_box in enumerate(objects['Car']):
        for other_box in objects['Car'][i + 1:]:
            if calculate_distance(car_box, other_box) < 100:
                if check_violation(f'car_approach_{i}', True, current_time):
                    violations[violation_labels['car_approach']] += 1
                    _draw_alert(frame, car_box, 'Car approach!')


def _check_pedestrian_violation(objects, violations, current_time, frame):
    for car_box in objects['Car']:
        for ped_box in objects['pedestrian']:
            if calculate_distance(car_box, ped_box) < 30:
                if check_violation('pedestrian', True, current_time):
                    violations[violation_labels['pedestrian']] += 1
                    _draw_alert(frame, car_box, 'Pedestrian close!')


def _check_red_light_violation(objects, violations, current_time, frame):
    for light_box in objects['Red traffic light']:
        for car_box in objects['Car']:
            if calculate_distance(light_box, car_box) < 150:
                if check_violation('red_light', True, current_time):
                    violations[violation_labels['red_light']] += 1
                    _draw_alert(frame, car_box, 'Red light!')


def _check_line_crossing(objects, solid_mask, violations, current_time, frame):
    for car_box in objects['Car']:
        x1, y1, x2, y2 = car_box
        car_mask_area = solid_mask[y1:y2, x1:x2]
        if np.any(car_mask_area > 0):
            if check_violation('line_crossing', True, current_time):
                violations[violation_labels['line_crossing']] += 1
                _draw_alert(frame, car_box, 'Line crossing!')


def _draw_alert(frame, box, text):
    x1, y1, x2, y2 = map(int, box)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
    cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)


def _draw_objects(frame, objects):
    for label, boxes in objects.items():
        for (x1, y1, x2, y2) in boxes:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
