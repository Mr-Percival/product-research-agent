"""Closed aspect taxonomy for whole appliances (ice makers, washers & dryers, range hoods, fridges, dishwashers).

Derived from data: open extraction on 1,000 held-in reviews, bge-small embeddings and agglomerative
clustering (scripts/discover_taxonomy.py, results/taxonomy_clusters.md). The clusters were named and merged by hand.
A closed set merges synonyms by construction ("noise", "loudness", "sound" -> noise).
"""

from __future__ import annotations

ASPECTS: dict[str, str] = {
    "core_performance": "How well it does its main job: cooling, washing, drying, dish cleaning, venting smoke.",
    "ice_production": "Ice maker output and speed: how much ice and how fast.",
    "ice_quality": "Ice cube size, shape, clarity, taste, melting, clumping.",
    "noise": "Noise, loudness, sound, vibration, rattling, humming.",
    "capacity_size": "Capacity, interior space, external dimensions, whether it fits the space or the items.",
    "ease_of_use": "Controls, settings, display, convenience, learning curve, day-to-day handling.",
    "installation_setup": "Installation, setup, assembly, instructions, hoses and hookups, venting install.",
    "cleaning_maintenance": "How easy the unit itself is to clean or maintain: filters, descaling, self-clean, mold.",
    "durability_reliability": "Durability, reliability, breaking down, lifespan, defects that appear over time.",
    "build_quality": "Materials and construction quality, flimsiness, finish, parts quality on arrival.",
    "appearance_design": "Looks, style, color, finish, lighting, design.",
    "price_value": "Price, cost, value for money, deal.",
    "efficiency": "Energy or water consumption, power draw, efficiency.",
    "temperature_control": "Temperature accuracy, range, consistency, thermostat.",
    "leaks_drainage": "Leaks, drainage, draining, water pooling, condensation.",
    "portability_weight": "Weight, portability, ease of moving it.",
    "customer_service_warranty": "Customer service, warranty, returns, refunds, replacements, repairs by the seller.",
    "delivery_packaging": "Shipping, delivery speed, packaging, condition on arrival.",
    "listing_accuracy": "Whether the product matches its description, photos or advertised specs.",
    "overall": "General satisfaction with no specific aspect ('great product', 'love it').",
}

# Aspects a shopping query can ask about. `overall` is excluded: it is not a feature.
QUERYABLE = [a for a in ASPECTS if a != "overall"]
