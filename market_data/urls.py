from django.urls import path
from market_data.views.pages import status_page, chart_page
from market_data.views.api import bars_json, assets_search

app_name = "market_data"
urlpatterns = [
    path("status/", status_page, name="status"),
    path("chart/", chart_page, name="chart"),
    path("api/bars/", bars_json, name="bars_json"),
    path("api/assets/", assets_search, name="assets_search"),
]
