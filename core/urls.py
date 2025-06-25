from django.urls import path
from . import views
from django.conf.urls.static import static
from django.conf import settings

urlpatterns = [
    path('', views.main, name='main'),
    path('upload', views.upload, name='upload'),
    path('progress/<int:video_id>/', views.get_progress, name='video_progress'),
    path('previous', views.previous, name='previous'),
    path('logout', views.logout, name='logout'),
    path('login', views.login, name='login'),
    path('register', views.register, name='register'),
    path('help', views.help, name='help'),
    path('result/<int:video_id>/', views.video_result, name='video_result'),
    path('graph/', views.violation_graph, name='violation_graph'),
    path('delete_selected/', views.delete_selected_videos, name='delete_selected'),
    path('delete_all/', views.delete_all_videos, name='delete_all'),
    path('detection', views.detection, name='detection'),
    path('detection/process/', views.detection_process, name='detection_process'),
    path('videos/<int:video_id>/update_title/', views.update_video_title, name='update_video_title'),
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)