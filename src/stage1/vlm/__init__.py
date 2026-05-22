"""Stage 1 VLM zero-shot inference (Table 1 row 4).

Uses original `OpenGVLab/InternVL3-2B` (not the OpenFACADES fine-tune, per
spec §3.2.1) with a self-written 6-field prompt to produce SFH/TH/MFH/AB
type plus year / period / floors / material / wwr per image, then aggregates
across the top-3 images per building with mode (categorical) / median
(numeric) + vote_share confidence.
"""
