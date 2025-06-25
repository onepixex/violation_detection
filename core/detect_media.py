from django.conf import settings
from ultralytics import YOLO
from pathlib import Path
import mimetypes
import subprocess

detection_model = YOLO(str(settings.YOLO_MODELS_DIR / 'detect_road_object.pt'))
segmentation_model = YOLO(str(settings.YOLO_MODELS_DIR / 'solid_line_seg.pt'))

def process_media(**kwargs):
    output_dir = Path(settings.MEDIA_ROOT) / 'processed' / 'detection'
    output_dir.mkdir(parents=True, exist_ok=True)

    source = kwargs.get('source')
    if not source:
        raise ValueError("Не указан источник 'source'")

    path = Path(source)
    is_video = mimetypes.guess_type(path)[0].startswith('video')

    conf = kwargs.get('conf', 0.5)
    iou = kwargs.get('iou', 0.7)
    max_det = kwargs.get('max_det', 100)
    save = kwargs.get('save', True)
    save_crop = kwargs.get('save_crop', False)
    line_width = kwargs.get('line_width', 2)
    show_labels = kwargs.get('show_labels', True)
    show_boxes = kwargs.get('show_boxes', True)
    vid_stride = kwargs.get('vid_stride', 2) if is_video else None

    raw_classes = kwargs.get('classes', [0, 1, 2])
    try:
        classes = [int(c) for c in raw_classes]
    except (ValueError, TypeError):
        classes = []

    valid_detection_classes = [0, 1, 2]
    detection_classes = [c for c in classes if c in valid_detection_classes]

    use_only_segmentation = set(classes) == {3}

    base_params = {
        "conf": conf,
        "iou": iou,
        "max_det": max_det,
        "save": save,
        "line_width": line_width,
        "show_labels": show_labels,
        "show_boxes": show_boxes,
        "project": output_dir,
        "name": "predict",
        "exist_ok": True,
    }

    if is_video:
        base_params["vid_stride"] = vid_stride
    else:
        base_params["save_crop"] = save_crop

    results = {}

    if use_only_segmentation:
        seg_params = base_params.copy()
        seg_params["source"] = str(path)
        results["segmentation"] = segmentation_model.predict(**seg_params)
        results["model_used"] = "segmentation_only"
        output_path = _get_latest_output_path(output_dir / "predict", path.name)
    else:
        det_params = base_params.copy()
        det_params["source"] = str(path)
        if detection_classes:
            det_params["classes"] = detection_classes

        det_results = detection_model.predict(**det_params)
        results["detection"] = det_results

        detect_img_path = _get_latest_output_path(output_dir / "predict", path.name)

        if 3 in classes and not is_video:
            seg_params = base_params.copy()
            seg_params["source"] = str(detect_img_path)
            seg_results = segmentation_model.predict(**seg_params)
            results["segmentation"] = seg_results

        output_path = _get_latest_output_path(output_dir / "predict", path.name)
        results["model_used"] = "combined"

    if is_video:
        output_path = _convert_video_to_mp4(output_path)

    return {
        "input_path": str(path),
        "output_path": str(output_path),
        "is_video": is_video,
        "used_params": base_params,
        "results": results
    }


def _get_latest_output_path(output_dir: Path, original_filename: str) -> Path:
    suffix = Path(original_filename).suffix.lower()

    if suffix in ['.mp4', '.avi', '.mov', '.mkv']:
        # Для видео: ищем файл .mp4 или .avi в output_dir
        video_files = sorted(
            [f for f in output_dir.glob("*") if f.suffix.lower() in ['.mp4', '.avi', '.mov', '.mkv']],
            key=lambda f: f.stat().st_mtime,
            reverse=True
        )
        if video_files:
            return video_files[0]

        # Если нет — ищем папку, в которую YOLO сохранил кадры
        subdirs = sorted(
            [f for f in output_dir.iterdir() if f.is_dir()],
            key=lambda f: f.stat().st_mtime,
            reverse=True
        )
        if subdirs:
            return subdirs[0]  # вернём путь к папке, потом используем для ffmpeg

        raise FileNotFoundError(f"Не найдены видеофайлы или директории в {output_dir}")

    # Для изображений
    img_suffixes = ['.jpg', '.jpeg', '.png']
    img_files = sorted(
        [f for f in output_dir.glob("*") if f.suffix.lower() in img_suffixes],
        key=lambda f: f.stat().st_mtime,
        reverse=True
    )
    if img_files:
        return img_files[0]

    raise FileNotFoundError(f"Файл с расширением {suffix} не найден в {output_dir}")


def _convert_video_to_mp4(input_path: Path) -> Path:
    # Если это папка — нужно собрать видео из изображений
    if input_path.is_dir():
        output_path = input_path.with_suffix('.mp4')
        frame_pattern = str(input_path / "%*.jpg")  # или .png — зависит от YOLO
        output_path = input_path.parent / f"{input_path.name}.mp4"

        command = [
            "ffmpeg",
            "-framerate", "25",
            "-pattern_type", "glob",
            "-i", str(input_path / "*.jpg"),
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            str(output_path)
        ]

        subprocess.run(command, check=True)
        return output_path

    # Если это уже файл
    if input_path.suffix.lower() == '.mp4':
        return input_path

    output_path = input_path.with_suffix('.mp4')
    command = [
        "ffmpeg",
        "-i", str(input_path),
        "-c:v", "libx264",
        "-preset", "slow",
        "-crf", "28",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        "-b:a", "128k",
        "-movflags", "+faststart",
        str(output_path)
    ]

    subprocess.run(command, check=True)
    input_path.unlink(missing_ok=True)

    return output_path

