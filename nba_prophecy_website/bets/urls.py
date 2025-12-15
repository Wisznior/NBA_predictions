from django.urls import path
from . import views

urlpatterns = [
    path("bet", views.bets_view, name="bet"),
    path("history", views.bets_view, name="bets_history")
]
