from django.urls import path
from . import views

urlpatterns = [
    path('', views.main, name='main'),
    path('home/', views.home, name='home'),  # Home page route
    path('video_feed_people_count/', views.video_feed_people_count, name='video_feed_people_count'),
    path('video_feed_object_detection/', views.video_feed_object_detection, name='video_feed_object_detection'),
]
