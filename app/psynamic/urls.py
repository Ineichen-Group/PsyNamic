
from django.urls import path
from django.template import loader

from . import views

urlpatterns = [
    path("", views.index, name="index"),
]