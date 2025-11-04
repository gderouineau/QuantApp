# strategies/urls.py

from django.urls import path
from strategies import views

app_name = "strategies"

urlpatterns = [
    path("custom/", views.CustomStrategyListView.as_view(), name="custom_list"),
    path("custom/new/", views.CustomStrategyCreateView.as_view(), name="custom_create"),
    path("custom/<int:pk>/edit/", views.CustomStrategyUpdateView.as_view(), name="custom_edit"),
    path("custom/<int:pk>/delete/", views.CustomStrategyDeleteView.as_view(), name="custom_delete"),
    path("custom/<int:pk>/test/", views.CustomStrategyTestView.as_view(), name="custom_test"),
]
