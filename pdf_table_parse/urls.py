from django.urls import path

from . import views

urlpatterns = [
    path('parse_pdf', views.index, name='index'),
    path('', views.hello, name='hello'),
]