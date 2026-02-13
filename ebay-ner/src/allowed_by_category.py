# allowed_by_category.py
# Determines whether a predicted aspect is allowed for a given category (recall-aware gating).
# High-impact aspects are always allowed (minimal gating), others must appear in category's allowed list.
# Category IDs here correspond to broad groups as per challenge annexure (e.g., 1 and 2).
# Define mapping of aspect names to categories where they are applicable.
ALLOWED_TAG_CATEGORIES = {
    "Anwendung": {2},
    "Farbe": {1},
    "Größe": {1, 2},
    "Hersteller": {1, 2},
    "Herstellernummer": {1, 2},
    "Herstellungsland_Und_-Region": {1},
    "Im_Lieferumfang_Enthalten": {1, 2},
    "Kompatible_Fahrzeug_Marke": {1, 2},
    "Kompatibles_Fahrzeug_Jahr": {1, 2},
    "Kompatibles_Fahrzeug_Modell": {1, 2},
    "Länge": {2},
    "Material": {1},
    "Maßeinheit": {1, 2},
    "Menge": {2},
    "Modell": {1, 2},
    "Oberflächenbeschaffenheit": {1},
    "Oe/Oem_Referenznummer(N)": {1, 2},
    "Produktart": {1, 2},
    "Produktlinie": {1},
    "SAE_Viskosität": {2},
    "Stärke": {1},
    "Technologie": {1},
    "Zähnezahl": {2}
}
# High-impact aspects to always allow (override category gating to boost recall).
HIGH_IMPACT_TAGS = {"Hersteller", "Produktart", "Kompatibles_Fahrzeug_Modell"}

def is_allowed(aspect_name: str, category_id: int) -> bool:
    """
    Check if an aspect prediction is allowed for a given category.
    High-impact aspects bypass category restrictions (always allowed).
    Returns True if the aspect is applicable to the category or is high-impact; False otherwise.
    """
    # Always allow high-impact aspects regardless of category (recall priority).
    if aspect_name in HIGH_IMPACT_TAGS:
        return True
    # Look up allowed categories for this aspect.
    allowed_cats = ALLOWED_TAG_CATEGORIES.get(aspect_name)
    if allowed_cats is None:
        # Aspect name not recognized (should not happen for known tags) - disallow by default.
        return False
    # Permit the tag if the category_id is in the aspect's allowed category set.
    return category_id in allowed_cats
