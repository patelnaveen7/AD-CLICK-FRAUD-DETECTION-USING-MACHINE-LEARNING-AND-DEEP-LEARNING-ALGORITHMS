from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('training/', views.training, name='training'),
    path('prediction/', views.prediction, name='prediction'),
]
