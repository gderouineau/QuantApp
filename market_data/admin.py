# market_data/admin.py
from __future__ import annotations

import json
import os
from typing import Any
from django.contrib import admin
from django.utils.html import format_html

from .models import Asset, DataFile, IngestionRun


# ---------- Helpers ----------
def human_size(num: int | float, suffix: str = "B") -> str:
    try:
        n = float(num)
    except Exception:
        return "—"
    for unit in ["", "K", "M", "G", "T", "P", "E", "Z"]:
        if abs(n) < 1024.0:
            return f"{n:.1f} {unit}{suffix}"
        n /= 1024.0
    return f"{n:.1f} Y{suffix}"


def ms_to_s(ms: int | float | None) -> str:
    if not ms and ms != 0:
        return "—"
    try:
        return f"{float(ms) / 1000.0:.3f}s"
    except Exception:
        return "—"


# ---------- Inlines ----------
class DataFileInline(admin.TabularInline):
    model = DataFile
    extra = 0
    can_delete = False
    fields = (
        "kind",
        "row_count",
        "last_date",
        "file_size_h",
        "updated_at",
        "path",
    )
    readonly_fields = (
        "kind",
        "row_count",
        "last_date",
        "file_size_h",
        "updated_at",
        "path",
    )

    def file_size_h(self, obj: DataFile) -> str:
        return human_size(obj.file_size)
    file_size_h.short_description = "Taille"


# ---------- ModelAdmins ----------
@admin.register(Asset)
class AssetAdmin(admin.ModelAdmin):
    list_display = (
        "symbol",
        "type",
        "exchange",
        "currency",
        "timezone",
        "is_active",
        "datafiles_count",
    )
    list_filter = ("type", "exchange", "currency", "is_active")
    search_fields = (
        "symbol",
        "exchange",
        "currency",
        "y_symbol",
        "av_symbol",
    )
    ordering = ("symbol",)
    actions = ("mark_active", "mark_inactive")
    inlines = [DataFileInline]
    list_per_page = 50

    def datafiles_count(self, obj: Asset) -> int:
        return obj.data_files.count()
    datafiles_count.short_description = "Fichiers"

    def mark_active(self, request, queryset):
        n = queryset.update(is_active=True)
        self.message_user(request, f"{n} asset(s) activé(s).")
    mark_active.short_description = "Activer les assets sélectionnés"

    def mark_inactive(self, request, queryset):
        n = queryset.update(is_active=False)
        self.message_user(request, f"{n} asset(s) désactivé(s).")
    mark_inactive.short_description = "Désactiver les assets sélectionnés"


@admin.register(DataFile)
class DataFileAdmin(admin.ModelAdmin):
    autocomplete_fields = ("asset",)
    list_display = (
        "asset",
        "kind",
        "row_count",
        "last_date",
        "file_size_h",
        "updated_at",
        "path_basename",
    )
    list_filter = ("kind", "updated_at")
    search_fields = ("asset__symbol", "path")
    readonly_fields = ("updated_at",)
    ordering = ("-updated_at",)
    list_per_page = 50

    fieldsets = (
        (None, {
            "fields": ("asset", "kind", "row_count", "last_date"),
        }),
        ("Fichier", {
            "fields": ("file_size", "file_size_h", "updated_at", "path"),
        }),
    )
    readonly_fields = ("file_size_h", "updated_at")

    def file_size_h(self, obj: DataFile) -> str:
        return human_size(obj.file_size)
    file_size_h.short_description = "Taille"

    def path_basename(self, obj: DataFile) -> str:
        return os.path.basename(obj.path or "") or "—"
    path_basename.short_description = "Fichier"


@admin.register(IngestionRun)
class IngestionRunAdmin(admin.ModelAdmin):
    list_display = (
        "id",
        "source",
        "timeframe",
        "started_at",
        "finished_at",
        "duration_s",
        "ok_count",
        "fail_count",
        "status_badge",
    )
    list_filter = ("source", "timeframe", "started_at", "finished_at")
    date_hierarchy = "started_at"
    ordering = ("-started_at",)
    list_per_page = 50

    # Présentation JSON jolie dans l'admin
    readonly_fields = (
        "started_at",
        "finished_at",
        "duration_ms",
        "duration_s",
        "ok_count",
        "fail_count",
        "anomalies_pretty",
        "errors_pretty",
    )

    fieldsets = (
        ("Run", {
            "fields": ("source", "timeframe", "started_at", "finished_at", "duration_ms", "duration_s"),
        }),
        ("Résultats", {
            "fields": ("ok_count", "fail_count"),
        }),
        ("Anomalies", {
            "fields": ("anomalies_pretty",),
        }),
        ("Erreurs", {
            "fields": ("errors_pretty",),
        }),
    )

    def duration_s(self, obj: IngestionRun) -> str:
        return ms_to_s(obj.duration_ms)
    duration_s.short_description = "Durée (s)"

    def status_badge(self, obj: IngestionRun) -> str:
        if obj.fail_count:
            color = "#b91c1c"  # rouge
            txt = "KO"
        elif obj.ok_count:
            color = "#15803d"  # vert
            txt = "OK"
        else:
            color = "#6b7280"  # gris
            txt = "—"
        return format_html(
            '<span style="background:{0};color:#fff;border-radius:9999px;padding:2px 8px;font-size:12px;">{1}</span>',
            color, txt
        )
    status_badge.short_description = "Statut"

    # Rendus JSON en <pre>
    def anomalies_pretty(self, obj: IngestionRun) -> Any:
        try:
            payload = json.dumps(obj.anomalies or [], ensure_ascii=False, indent=2)
        except Exception:
            payload = "—"
        return format_html('<pre style="max-width:100%;white-space:pre-wrap;">{}</pre>', payload)
    anomalies_pretty.short_description = "Anomalies"

    def errors_pretty(self, obj: IngestionRun) -> Any:
        try:
            payload = json.dumps(obj.errors or [], ensure_ascii=False, indent=2)
        except Exception:
            payload = "—"
        return format_html('<pre style="max-width:100%;white-space:pre-wrap;">{}</pre>', payload)
    errors_pretty.short_description = "Erreurs"
