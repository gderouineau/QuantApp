from django.db import models
from django.utils import timezone
from django.core.validators import MinValueValidator, MaxValueValidator

class NSUniverse(models.Model):
    """
    Décrit un univers + UT (ex: FR-PEA / DAILY).
    Tu peux créer une instance 'FR-PEA' et y associer timeframe='1D'.
    """
    name = models.CharField(max_length=100, unique=True)
    code = models.CharField(max_length=50, unique=True)  # ex: "FR_PEA"
    timeframe = models.CharField(max_length=16, default="1D")  # "15m","1h","1D","1W"
    asset_class = models.CharField(max_length=32, default="equities")  # equities, index, forex, etc.
    is_active = models.BooleanField(default=True)

    asset_group = models.ForeignKey(
        "portfolios.AssetGroup",
        on_delete=models.SET_NULL,
        null=True, blank=True,
        related_name="ns_universes",
    )
    # Fallback si tu veux juste mettre un nom de groupe sans FK
    asset_group_name = models.CharField(max_length=128, blank=True, default="")


    def __str__(self):
        return f"{self.name} [{self.timeframe}]"


class NSDataset(models.Model):
    """
    Décrit une config de dataset + artefacts produits (labels, features).
    On peut réutiliser la même config pour générer labels/dataset sur des périodes différentes.
    """
    SCALING_MODES = [
        ("grid_scaled", "Grid scaled (position 0..Y selon min/max de ref + indicateurs)"),
        ("relative_feats", "Relative features (ratios invariants d'échelle)"),
    ]

    universe = models.ForeignKey(NSUniverse, on_delete=models.PROTECT, related_name="datasets")
    date_from = models.DateField(null=True, blank=True)
    date_to = models.DateField(null=True, blank=True)

    x_window = models.PositiveIntegerField(default=1, validators=[MinValueValidator(1)])
    y_resolution = models.PositiveIntegerField(default=500, validators=[MinValueValidator(50)])
    horizon_bars = models.PositiveIntegerField(default=20, validators=[MinValueValidator(1)])

    TF = [("1D", "Daily"), ("1H", "Hourly")]
    timeframe = models.CharField(max_length=2, choices=TF, default="1D")

    # Marges TP (ex: [0.10, 0.15]) stockées en JSON
    tp_margins = models.JSONField(default=list, help_text="Liste de marges TP (ex: [0.10, 0.15])")

    # Indicateurs (activables au fil du temps)
    use_ma = models.BooleanField(default=True)
    ma_periods = models.JSONField(default=list, help_text="Ex: [20, 50]")
    use_bb = models.BooleanField(default=True)
    bb_period = models.PositiveIntegerField(default=20)
    bb_k = models.FloatField(default=2.0)
    use_atr = models.BooleanField(default=True)
    atr_period = models.PositiveIntegerField(default=14)

    scaling_mode = models.CharField(max_length=32, choices=SCALING_MODES, default="grid_scaled")

    # Artefacts (chemins vers fichiers parquet/csv/npz)
    labels_path = models.CharField(max_length=255, blank=True, default="")
    dataset_path = models.CharField(max_length=255, blank=True, default="")
    manifest_path = models.CharField(max_length=255, blank=True, default="")

    created_at = models.DateTimeField(auto_now_add=True)
    last_built_at = models.DateTimeField(null=True, blank=True)

    def __str__(self):
        return f"Dataset {self.pk} - {self.universe.code} [{self.universe.timeframe}]"


class NSRun(models.Model):
    """
    Journalise les exécutions (labels/dataset/backtest).
    """
    KIND = [
        ("labels", "Build Labels"),
        ("dataset", "Build Dataset"),
        ("backtest", "Backtest/Validation"),
        ("train", "Train (ML)"),
    ]
    STATUS = [
        ("pending", "Pending"),
        ("running", "Running"),
        ("done", "Done"),
        ("failed", "Failed"),
    ]

    dataset = models.ForeignKey(NSDataset, on_delete=models.CASCADE, related_name="runs")
    kind = models.CharField(max_length=16, choices=KIND)
    started_at = models.DateTimeField(default=timezone.now)
    finished_at = models.DateTimeField(null=True, blank=True)
    status = models.CharField(max_length=16, choices=STATUS, default="pending")
    log = models.TextField(blank=True, default="")
    metrics = models.JSONField(default=dict, blank=True)

    def mark_running(self):
        self.status = "running"
        self.started_at = timezone.now()
        self.save(update_fields=["status", "started_at"])

    def mark_done(self, metrics=None):
        self.status = "done"
        self.finished_at = timezone.now()
        if metrics:
            self.metrics = metrics
        self.save(update_fields=["status", "finished_at", "metrics"])

    def mark_failed(self, error_msg):
        self.status = "failed"
        self.finished_at = timezone.now()
        self.log = (self.log or "") + f"\nERROR: {error_msg}"
        self.save(update_fields=["status", "finished_at", "log"])


class NSPrediction(models.Model):
    """
    Stocke des proba/décisions par actif/date pour analyse/backtest.
    Label réel optionnel si connu après coup.
    """
    dataset = models.ForeignKey(NSDataset, on_delete=models.CASCADE, related_name="predictions")
    instrument = models.CharField(max_length=50)
    ts = models.DateTimeField()  # fin de barre t
    prob_tp10 = models.FloatField(null=True, blank=True, validators=[MinValueValidator(0.0), MaxValueValidator(1.0)])
    prob_tp15 = models.FloatField(null=True, blank=True, validators=[MinValueValidator(0.0), MaxValueValidator(1.0)])
    decision = models.BooleanField(default=False)  # selon un seuil choisi
    label_tp10 = models.BooleanField(null=True, blank=True)  # 1 si TP10 avant SL, 0 sinon
    label_tp15 = models.BooleanField(null=True, blank=True)
    meta = models.JSONField(default=dict, blank=True)

    class Meta:
        indexes = [
            models.Index(fields=["dataset", "instrument", "ts"]),
        ]
        unique_together = ("dataset", "instrument", "ts")
