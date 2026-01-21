from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('process/', views.process_audio, name='process'),
    path('train/', views.train_page, name='train'),
    path('train_setup/', views.setup_training_data, name='train_setup'),
]