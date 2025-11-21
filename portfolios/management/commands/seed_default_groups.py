# portfolios/management/commands/seed_default_groups.py
from django.core.management.base import BaseCommand
from django.contrib.auth import get_user_model

from market_data.models import Asset
from portfolios.models import AssetGroup

User = get_user_model()

SETS = {
    "US Mega": ["AAPL", "MSFT", "GOOGL", "AMZN", "META"],
    "CAC40 (échantillon)": ["AIR.PA","AI.PA","BNP.PA","CAP.PA","CA.PA","DG.PA","DSY.PA","ENGI.PA","KER.PA","LR.PA","MC.PA","ORA.PA","OR.PA","RMS.PA","RI.PA","SAF.PA","SAN.PA","SGO.PA","SU.PA","TTE.PA"],
    "FX Majors": ["EURUSD","GBPUSD","USDJPY","USDCHF","AUDUSD","USDCAD"],
}

class Command(BaseCommand):
    help = "Crée des groupes d'actifs globaux de base (si les symboles existent déjà)."

    def add_arguments(self, parser):
        parser.add_argument("--force", action="store_true", help="Réécrit la composition des groupes")

    def handle(self, *args, **opts):
        force = opts["force"]
        total = 0

        for name, symbols in SETS.items():
            grp, _ = AssetGroup.objects.get_or_create(name=name, is_global=True, defaults={"user": None})
            qs = Asset.objects.filter(symbol__in=symbols)
            if not qs.exists():
                self.stdout.write(self.style.WARNING(f"[{name}] aucun symbole trouvé, crée d'abord les assets."))
                continue
            if force:
                grp.assets.set(qs)
            else:
                # Ajoute ce qui manque sans retirer l'existant
                existing = set(grp.assets.values_list("symbol", flat=True))
                to_add = qs.exclude(symbol__in=existing)
                if to_add.exists():
                    grp.assets.add(*list(to_add))
            total += grp.assets.count()
            self.stdout.write(self.style.SUCCESS(f"[{name}] ok — {grp.assets.count()} actifs."))

        self.stdout.write(self.style.SUCCESS(f"Terminé. Total liens groupe-actif ~ {total}."))
