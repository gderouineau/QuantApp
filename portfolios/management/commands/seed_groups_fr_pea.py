# portfolios/management/commands/seed_groups_fr_pea.py
from __future__ import annotations
from pathlib import Path

from django.core.management.base import BaseCommand
from django.contrib.auth import get_user_model

from market_data.models import Asset
from portfolios.models import AssetGroup

User = get_user_model()

def _read_list(p: Path) -> list[str]:
    if not p.exists():
        return []
    return [ln.strip() for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]

class Command(BaseCommand):
    help = "Crée/alimente les groupes globaux 'FR — PEA' et 'FR — PEA-PME' depuis data/universes/*.txt"

    def add_arguments(self, parser):
        parser.add_argument("--base", default="data/universes", help="Dossier contenant PEA_fr.txt et PEA_PME_fr.txt")
        parser.add_argument("--force", action="store_true", help="Remplace la composition au lieu d'ajouter ce qui manque")

    def handle(self, *args, **opts):
        base = Path(opts["base"]).expanduser()
        pea_file = base / "PEA_fr.txt"
        pme_file = base / "PEA_PME_fr.txt"

        pea_syms = _read_list(pea_file)
        pme_syms = _read_list(pme_file)

        grp_pea, _ = AssetGroup.objects.get_or_create(name="FR — PEA", is_global=True, defaults={"user": None})
        grp_pme, _ = AssetGroup.objects.get_or_create(name="FR — PEA-PME", is_global=True, defaults={"user": None})

        def sync_group(grp: AssetGroup, syms: list[str], label: str):
            qs = Asset.objects.filter(symbol__in=syms)
            if not qs.exists():
                self.stdout.write(self.style.WARNING(f"[{label}] aucun symbole trouvé dans Asset (as-tu importé les assets ?)"))
                return
            if opts["force"]:
                grp.assets.set(qs)
            else:
                existing = set(grp.assets.values_list("symbol", flat=True))
                to_add = qs.exclude(symbol__in=existing)
                if to_add.exists():
                    grp.assets.add(*list(to_add))
            self.stdout.write(self.style.SUCCESS(f"[{label}] {grp.assets.count()} actifs dans le groupe."))

        if pea_syms:
            sync_group(grp_pea, pea_syms, "FR — PEA")
        if pme_syms:
            sync_group(grp_pme, pme_syms, "FR — PEA-PME")

        self.stdout.write(self.style.SUCCESS("Terminé."))
