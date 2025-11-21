from django.urls import path
from . import views

app_name = "portfolios"

urlpatterns = [
    # Portefeuilles
    path("", views.PortfolioListView.as_view(), name="list"),
    path("new/", views.PortfolioCreateView.as_view(), name="create"),
    path("<int:pk>/", views.PortfolioDetailView.as_view(), name="detail"),
    path("<int:pk>/edit/", views.PortfolioUpdateView.as_view(), name="edit"),
    path("<int:pk>/delete/", views.PortfolioDeleteView.as_view(), name="delete"),

    # Allocations
    path("<int:portfolio_id>/allocations/new/", views.AllocationCreateView.as_view(), name="alloc_create"),
    path("allocations/<int:pk>/edit/", views.AllocationUpdateView.as_view(), name="alloc_edit"),
    path("allocations/<int:pk>/delete/", views.AllocationDeleteView.as_view(), name="alloc_delete"),

    # Groupes d'actifs
    path("groups/", views.GroupListView.as_view(), name="groups"),
    path("groups/new/", views.GroupCreateView.as_view(), name="group_create"),
    path("groups/<int:pk>/edit/", views.GroupUpdateView.as_view(), name="group_edit"),
    path("groups/<int:pk>/delete/", views.GroupDeleteView.as_view(), name="group_delete"),

    # Simulation (depuis UI)
    path("<int:pk>/simulate/", views.PortfolioSimulateView.as_view(), name="simulate"),
    path("runs/<int:pk>/", views.PortfolioRunDetailView.as_view(), name="run_detail"),


    path("<int:pk>/inbox/", views.PortfolioSignalInboxView.as_view(), name="inbox"),
    path("<int:pk>/accept-signal/", views.PortfolioAcceptSignalView.as_view(), name="accept_signal"),

    path("positions/<int:pk>/", views.PositionDetailView.as_view(), name="position_detail"),
    path("positions/<int:pk>/move-sl/", views.PositionMoveSLView.as_view(), name="position_move_sl"),
    path("positions/<int:pk>/partial-exit/", views.PositionPartialExitView.as_view(), name="position_partial_exit"),
    path("positions/<int:pk>/close/", views.PositionCloseView.as_view(), name="position_close"),


    path("<int:pk>/backtest/", views.PortfolioBacktestView.as_view(), name="backtest"),
    path("runs/<int:pk>/", views.PortfolioRunDetailView.as_view(), name="run_detail"),


]
