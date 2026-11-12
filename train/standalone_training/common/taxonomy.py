from pathlib import Path
from typing import Dict, Optional, Any
import yaml

# Lazy-loaded cache
_SPECIES_CACHE: Optional[Dict[str, Dict[str, Any]]] = None
_FAMILIES_CACHE: Optional[Dict[str, Dict[str, Any]]] = None


def _load_data():
    global _SPECIES_CACHE, _FAMILIES_CACHE
    if _SPECIES_CACHE is not None and _FAMILIES_CACHE is not None:
        return

    base_dir = Path(__file__).resolve().parent.parent
    species_path = base_dir / "species.yaml"
    families_path = base_dir / "taxonomy_families.yaml"

    with open(species_path, "r", encoding="utf-8") as f:
        species_data = yaml.safe_load(f).get("species", [])
        _SPECIES_CACHE = {item["id"]: item for item in species_data}

    with open(families_path, "r", encoding="utf-8") as f:
        families_data = yaml.safe_load(f).get("families", [])
        _FAMILIES_CACHE = {item["id"]: item for item in families_data}


def species_to_family(species_id: str) -> Optional[Dict[str, str]]:
    """
    Возвращает {'family_id': 'F01', 'latin_name': 'Poaceae', 'ru_name': 'Злаковые'};
    None для неизвестного вида.
    Строго по контракту ТЗ § 5.
    """
    if not isinstance(species_id, str):
        return None

    clean_id = species_id.strip()
    _load_data()

    if clean_id not in _SPECIES_CACHE:
        return None

    sp = _SPECIES_CACHE[clean_id]
    family_id = sp.get("family_id")
    if not family_id or family_id not in _FAMILIES_CACHE:
        return None

    fam = _FAMILIES_CACHE[family_id]
    return {
        "family_id": fam["id"],
        "latin_name": fam["latin_name"],
        "ru_name": fam["ru_name"]
    }


def lookup_species_by_scientific_name(name: str) -> Optional[Dict[str, Any]]:
    """Поиск вида по латинскому названию (напр. 'Zea mays')."""
    if not isinstance(name, str):
        return None
    clean_name = name.strip().lower()
    _load_data()
    for sp in _SPECIES_CACHE.values():
        if sp["latin_name"].strip().lower() == clean_name:
            return sp
    return None

