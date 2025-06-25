from django.db import models
from django.contrib.auth.models import User

class AlgorithmVideo(models.Model):
    user = models.ForeignKey(User, on_delete=models.SET_NULL, null=True, blank=True)
    session_key = models.CharField(max_length=40, null=True, blank=True)
    title = models.CharField(max_length=255, blank=True, default="")
    video = models.FileField(upload_to='uploads/video/')
    processed_video = models.FileField(upload_to='processed/video/', null=True, blank=True)
    violation = models.JSONField(default=dict, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.title or f"Видео #{self.pk}"

class RecognitionModel(models.Model):
    MEDIA_TYPE_CHOICES = (
        ('image', 'Image'),
        ('video', 'Video'),
    )
    user = models.ForeignKey(User, on_delete=models.SET_NULL, null=True, blank=True)
    session_key = models.CharField(max_length=40, null=True, blank=True)
    title = models.CharField(max_length=255, blank=True, default="", verbose_name="Название")  # Новое поле
    original_media = models.FileField(upload_to='uploads/recognition/')
    processed_media = models.FileField(upload_to='processed/recognition/', null=True, blank=True)
    media_type = models.CharField(max_length=10, choices=MEDIA_TYPE_CHOICES)
    media_name = models.CharField(max_length=255, null=True, blank=True)
    processing_params = models.JSONField(default=dict)
    detection_txt = models.FileField(upload_to='results/txt/', null=True, blank=True)
    cropped_images = models.JSONField(default=list, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        title_display = self.title or f"Объект {self.pk}"
        return f"{title_display} ({self.user or 'unreg'})"