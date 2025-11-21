from django.contrib import admin
from .models import NSUniverse, NSDataset, NSRun, NSPrediction

@admin.register(NSUniverse)
class NSUniverseAdmin(admin.ModelAdmin):
    list_display = ("name", "code", "timeframe", "asset_class", "is_active")
    list_filter = ("timeframe", "asset_class", "is_active")
    search_fields = ("name", "code")

@admin.register(NSDataset)
class NSDatasetAdmin(admin.ModelAdmin):
    list_display = ("id", "universe", "date_from", "date_to", "x_window", "y_resolution",
                    "horizon_bars", "scaling_mode", "last_built_at")
    list_filter = ("universe", "scaling_mode")
    search_fields = ("id", "labels_path", "dataset_path")
    readonly_fields = ("last_built_at",)

@admin.register(NSRun)
class NSRunAdmin(admin.ModelAdmin):
    list_display = ("id", "dataset", "kind", "status", "started_at", "finished_at")
    list_filter = ("kind", "status")
    search_fields = ("id", "dataset__id")

@admin.register(NSPrediction)
class NSPredictionAdmin(admin.ModelAdmin):
    list_display = ("dataset", "instrument", "ts", "prob_tp10", "prob_tp15", "decision")
    list_filter = ("dataset", "instrument")
    search_fields = ("instrument",)
