from django.urls import path
from . import views

urlpatterns = [
    path("future", views.future, name="future"),
    path("history", views.history, name="history"),
]
