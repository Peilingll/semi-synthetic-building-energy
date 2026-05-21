"""Stage 1: visual attribute extraction accuracy (spec §3.4.1).

Predicts building_type / bouwjaar / num_floors from street-view images.
Phase A: Delft + DINOv2 frozen + MLP, 1 fold smoke-test.
Phase B: extend to 4 cities, 5-fold + LOCO, add ResNet-50 / Swin-T / Gemini.
"""
