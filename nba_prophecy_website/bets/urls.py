from django.urls import path
from . import views

urlpatterns = [
    path("", views.bets_view, name="bets"),
]
