# neuronal_strategy/management/commands/ns_make_dataset_from_group.py
from django.core.management.base import BaseCommand
from neuronal_strategy.models import NSUniverse, NSDataset
from portfolios.models import AssetGroup

class Command(BaseCommand):
    help = "Crée NSUniverse + NSDataset à partir d'un AssetGroup."

    def add_arguments(self, parser):
        parser.add_argument("--group", required=True, help='Nom exact du groupe (ex: "FR — PEA")')
        parser.add_argument("--timeframe", default="1D", choices=["1H", "1D", "1W"])
        parser.add_argument("--date_from", default=None)
        parser.add_argument("--date_to", default=None)
        parser.add_argument("--y", type=int, default=256)  # <= recommandé
        parser.add_argument("--h", type=int, default=30)   # horizon barres par défaut
        parser.add_argument("--tp", default="0.15")        # marge TP par défaut
        parser.add_argument("--scaling", default="grid_scaled", choices=["grid_scaled","relative_feats"])

    def handle(self, *args, **opts):
        grp = AssetGroup.objects.get(name=opts["group"])
        tf = opts["timeframe"]
        uname = f"{grp.name} {tf}"
        ucode = f"{grp.id}_{tf}"

        u, _ = NSUniverse.objects.get_or_create(
            code=ucode,
            defaults=dict(name=uname, timeframe=tf, asset_class="equities", asset_group=grp),
        )

        tp_margins = [float(x.strip()) for x in opts["tp"].split(",") if x.strip()]

        ds = NSDataset.objects.create(
            universe=u,
            timeframe=tf,
            date_from=opts["date_from"],
            date_to=opts["date_to"],
            x_window=1,
            y_resolution=opts["y"],
            horizon_bars=opts["h"],
            tp_margins=tp_margins,
            scaling_mode=opts["scaling"],
            use_ma=True,  ma_periods=[20, 50],  # MM20/50 dispo
            use_bb=True,  bb_period=20, bb_k=2.0,
            use_atr=True, atr_period=14,
        )
        self.stdout.write(self.style.SUCCESS(f"Universe={u.id} Dataset={ds.id}"))
