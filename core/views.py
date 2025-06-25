import os
import threading
from pathlib import Path
from uuid import uuid4
from django.shortcuts import render, redirect
from django.http import JsonResponse, HttpResponse, FileResponse
from django.conf import settings
from django.core.files.base import ContentFile
from django.utils.crypto import get_random_string
from matplotlib import pyplot as plt
import numpy as np

from core.models import AlgorithmVideo, RecognitionModel
from core.violation_detect_alg import process_video

from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure

from django.contrib.auth import authenticate, login as auth_login, logout as auth_logout
from django.contrib.auth.models import User
from django.contrib import messages

from collections import Counter
from itertools import chain

import textwrap
import traceback

from django.views.decorators.http import require_POST
from django.views.decorators.csrf import csrf_exempt
import json

from core.detect_media import process_media
from mimetypes import guess_type

VIDEO_PROGRESS = {}
def main(request):
    return render(request, 'core/main.html')

def previous(request):
    if request.user.is_authenticated:
        videos = AlgorithmVideo.objects.filter(user=request.user)
        recognition_models = RecognitionModel.objects.filter(user=request.user)
    else:
        videos = AlgorithmVideo.objects.filter(session_key=request.session.session_key)
        recognition_models = RecognitionModel.objects.filter(session_key=request.session.session_key)

    return render(request, 'core/previous.html', {
        'videos': videos,
        'recognition_models': recognition_models
    })

def login(request):
    if request.method == 'POST':
        username = request.POST.get('username')
        password = request.POST.get('password')
        user = authenticate(request, username=username, password=password)

        if user:
            auth_login(request, user)
            return redirect('main')
        else:
            messages.error(request, 'Неверный логин или пароль.')

    return render(request, 'core/login.html')


def register(request):
    if request.method == 'POST':
        username = request.POST.get('username')
        password = request.POST.get('password')
        confirm = request.POST.get('confirm')

        if password != confirm:
            messages.error(request, 'Пароли не совпадают.')
        elif User.objects.filter(username=username).exists():
            messages.error(request, 'Пользователь с таким именем уже существует.')
        else:
            User.objects.create_user(username=username, password=password)
            messages.success(request, 'Регистрация успешна. Войдите в систему.')
            return redirect('login')

    return render(request, 'core/register.html')


def logout(request):
    auth_logout(request)
    return redirect('main')


def upload(request):
    if request.method == 'POST' and request.FILES.get('video'):
        video_file = request.FILES['video']

        # Гарантированно создаём и получаем session_key
        if not request.session.session_key:
            request.session.save()
        session_key = request.session.session_key

        # Получаем и обрабатываем заголовок
        raw_title = request.POST.get('title', '').strip()
        title = raw_title or Path(video_file.name).stem

        # Сохраняем видео в БД
        algo_video = AlgorithmVideo.objects.create(
            video=video_file,
            title=title,
            user=request.user if request.user.is_authenticated else None,
            session_key=session_key
        )

        # Создаём временный путь для оригинального видео
        temp_video_path = os.path.join(settings.MEDIA_ROOT, f'temp_{uuid4()}.mp4')
        with open(temp_video_path, 'wb') as f:
            for chunk in video_file.chunks():
                f.write(chunk)

        # Прогресс обработки
        def progress_callback(progress):
            VIDEO_PROGRESS[algo_video.id] = progress

        # Фоновая обработка видео
        def process_and_save():
            try:
                output_path, violations = process_video(
                    temp_video_path,
                    settings.MEDIA_ROOT,
                    progress_callback
                )
                with open(output_path, 'rb') as f:
                    algo_video.processed_video.save(
                        os.path.basename(output_path),
                        ContentFile(f.read())
                    )
                algo_video.violation = violations
                algo_video.save()
                VIDEO_PROGRESS[algo_video.id] = 100
            except Exception as e:
                VIDEO_PROGRESS[algo_video.id] = -1  # Для ошибок можно использовать -1
                print(f"Ошибка при обработке видео {algo_video.id}:", e)
            finally:
                # Удаление временных файлов
                try:
                    if os.path.exists(temp_video_path):
                        os.remove(temp_video_path)
                    if 'output_path' in locals() and os.path.exists(output_path):
                        os.remove(output_path)
                except Exception as cleanup_error:
                    print("Ошибка при удалении временных файлов:", cleanup_error)

        threading.Thread(target=process_and_save).start()

        return JsonResponse({'video_id': algo_video.id})

    return render(request, 'core/upload.html')


def help(request):
    return render(request, 'core/help.html')

def get_progress(request, video_id):
    progress = VIDEO_PROGRESS.get(int(video_id), 0)
    return JsonResponse({'progress': progress})

def video_result(request, video_id):
    try:
        video = AlgorithmVideo.objects.get(id=video_id)
        return JsonResponse({'video_url': video.processed_video.url if video.processed_video else None})
    except AlgorithmVideo.DoesNotExist:
        return JsonResponse({'error': 'not found'}, status=404)


def violation_graph(request):
    video_ids = request.GET.getlist('video_ids')
    mode = request.GET.get('mode', 'single')

    videos = AlgorithmVideo.objects.filter(id__in=video_ids).order_by('created_at')

    fig = Figure(figsize=(14, 8))
    ax = fig.subplots()

    # Увеличиваем базовый размер шрифта для всех элементов
    plt.rcParams.update({'font.size': 14})

    if mode == 'compare':
        if len(videos) < 2:
            ax.text(0.5, 0.5, 'Для сравнения необходимо минимум 2 видео',
                    ha='center', va='center', fontsize=16)  # Увеличенный шрифт
        else:
            all_violations = set()
            for video in videos:
                if video.violation:
                    all_violations.update(video.violation.keys())
            all_violations = sorted(all_violations)

            data = []
            for video in videos:
                video_data = {
                    'title': video.title or f"Видео {video.id}",
                    'violations': []
                }
                for violation in all_violations:
                    count = video.violation.get(violation, 0) if video.violation else 0
                    video_data['violations'].append(count)
                data.append(video_data)

            bar_width = 0.8 / len(data)
            colors = ['blue', 'orange', 'green', 'red', 'yellow', 'purple']

            for i, video in enumerate(data):
                positions = [x + i * bar_width for x in range(len(all_violations))]
                ax.bar(positions, video['violations'],
                       width=bar_width,
                       color=colors[i % len(colors)],
                       label=video['title'],
                       edgecolor='black',
                       linewidth=0.5)

            wrapped_labels = ['\n'.join(textwrap.wrap(label, 12)) for label in all_violations]
            ax.set_xticks([x + (len(data) - 1) * bar_width / 2 for x in range(len(all_violations))])
            ax.set_xticklabels(wrapped_labels, ha='right', fontsize=14)  # Увеличенный шрифт
            ax.legend(bbox_to_anchor=(0.85, 1), loc='upper left', fontsize=12)  # Увеличенный шрифт
            ax.set_title('Сравнение нарушений по выбранным видео', pad=20, fontsize=16)  # Увеличенный шрифт
            ax.set_ylabel('Количество нарушений', fontsize=14)  # Увеличенный шрифт
            ax.tick_params(axis='y', labelsize=12)  # Увеличенный шрифт для меток оси Y

    else:
        all_violations = list(chain.from_iterable(
            v.violation.items() for v in videos if v.violation
        ))
        counter = Counter()
        for key, count in all_violations:
            counter[key] += count

        labels = list(counter.keys())
        values = list(counter.values())

        wrapped_labels = ['\n'.join(textwrap.wrap(label, 12)) for label in labels]
        colors = plt.cm.Blues([0.4 + 0.6*i/len(labels) for i in range(len(labels))])
        ax.bar(wrapped_labels, values, color=colors, edgecolor='black', linewidth=0.5)
        ax.set_title('Сводная статистика нарушений' if len(videos) > 1 else 'Статистика нарушений',
                    fontsize=16)  # Увеличенный шрифт
        ax.set_ylabel('Количество нарушений', fontsize=14)  # Увеличенный шрифт
        ax.tick_params(axis='x', labelsize=12)  # Увеличенный шрифт для меток оси X
        ax.tick_params(axis='y', labelsize=12)  # Увеличенный шрифт для меток оси Y

    plt.tight_layout()
    canvas = FigureCanvas(fig)
    response = HttpResponse(content_type='image/png')
    canvas.print_png(response)
    return response

@require_POST
@csrf_exempt
def delete_selected_videos(request):
    try:
        data = json.loads(request.body)
        ids = data.get("video_ids", [])
        if request.user.is_authenticated:
            AlgorithmVideo.objects.filter(id__in=ids, user=request.user).delete()
        else:
            AlgorithmVideo.objects.filter(id__in=ids, session_key=request.session.session_key).delete()
        return JsonResponse({"status": "ok"})
    except Exception as e:
        return JsonResponse({"status": "error", "message": str(e)}, status=400)

@require_POST
@csrf_exempt
def delete_all_videos(request):
    try:
        if request.user.is_authenticated:
            AlgorithmVideo.objects.filter(user=request.user).delete()
        else:
            AlgorithmVideo.objects.filter(session_key=request.session.session_key).delete()
        return JsonResponse({"status": "ok"})
    except Exception as e:
        return JsonResponse({"status": "error", "message": str(e)}, status=400)


def detection(request):
    return render(request, 'core/detection.html')

@require_POST
@csrf_exempt
def detection_process(request):
    try:
        uploaded_file = request.FILES.get('media')
        if not uploaded_file:
            return JsonResponse({'error': 'Файл не был загружен'}, status=400)

        custom_name = request.POST.get('custom_name') or Path(uploaded_file.name).stem
        ext = Path(uploaded_file.name).suffix
        filename = f"{custom_name}_{get_random_string(6)}{ext}"

        title = custom_name[:255]

        original_path = Path(settings.MEDIA_ROOT) / 'uploads' / 'recognition' / filename
        original_path.parent.mkdir(parents=True, exist_ok=True)

        with open(original_path, 'wb+') as dest:
            for chunk in uploaded_file.chunks():
                dest.write(chunk)

        try:
            class_list = json.loads(request.POST.get('class_list', '[]'))
        except json.JSONDecodeError:
            class_list = []

        params = {
            'source': str(original_path),
            'conf': float(request.POST.get('conf', 0.5)),
            'iou': float(request.POST.get('iou', 0.7)),
            'max_det': int(request.POST.get('max_det', 100)),
            'save': True,
            'line_width': int(request.POST.get('line_width', 2)),
            'show_labels': request.POST.get('show_labels') == 'on',
            'show_boxes': request.POST.get('show_boxes') == 'on',
            'vid_stride': int(request.POST.get('vid_stride', 1)),
            'classes': class_list,
        }

        result = process_media(**params)

        output_path = Path(result["output_path"])

        if not output_path.exists():
            raise FileNotFoundError(f"Результирующий файл не найден: {output_path}")

        if not request.session.session_key:
            request.session.create()

        recognition = RecognitionModel.objects.create(
            title=title,
            original_media=f"uploads/recognition/{filename}",
            processed_media=str(output_path).replace(str(settings.MEDIA_ROOT) + os.sep, ''),
            media_type='video' if output_path.suffix.lower() in ['.mp4', '.avi', '.mov', '.mkv'] else 'image',
            processing_params=params,
            user=request.user if request.user.is_authenticated else None,
            session_key=request.session.session_key,
            media_name=filename
        )

        return JsonResponse({
            'output_url': settings.MEDIA_URL + recognition.processed_media.name,
            'media_type': recognition.media_type,
            'recognition_id': recognition.id,
        })

    except Exception as e:
        traceback.print_exc()
        return JsonResponse({'error': str(e)}, status=500)

def serve_processed_video(request, video_id):
    try:
        video = AlgorithmVideo.objects.get(id=video_id)
        if not video.processed_video:
            return JsonResponse({'error': 'Видео не обработано'}, status=404)

        video_path = video.processed_video.path
        content_type, _ = guess_type(video_path)
        return FileResponse(open(video_path, 'rb'), content_type=content_type or 'video/mp4')

    except AlgorithmVideo.DoesNotExist:
        return JsonResponse({'error': 'Видео не найдено'}, status=404)
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)

@csrf_exempt
def update_video_title(request, video_id):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            title = data.get('title', '').strip()
            if not title:
                return JsonResponse({'error': 'Пустое название'}, status=400)

            video = AlgorithmVideo.objects.get(id=video_id)
            video.title = title
            video.save()
            return JsonResponse({'message': 'Название обновлено'})
        except AlgorithmVideo.DoesNotExist:
            return JsonResponse({'error': 'Видео не найдено'}, status=404)
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)
    return JsonResponse({'error': 'Недопустимый метод'}, status=405)