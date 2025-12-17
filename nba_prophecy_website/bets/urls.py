from django.urls import path
from . import views

urlpatterns = [
    path("", views.bets_view, name="bet"),
    path("history/", views.bets_history_view, name="bets_history"),
]
