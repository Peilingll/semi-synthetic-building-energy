# Table 3 — Stage 1 paradigm comparison, LOCO-amsterdam

Trained paradigms: trained on R+U+D, evaluated on amsterdam.
Zero-shot paradigms: pooled predictions reused (split-independent).

## Full LOCO hold-out (all imaged amsterdam buildings)

| model | n | type_acc | type_macro_f1 | year_mae | period_acc | floors_mae |
|---|---:|---:|---:|---:|---:|---:|
| DINOv2 frozen | 8011 | 0.8969 | 0.3726 | 19.50 | 0.8760 | 0.574 |
| ResNet-50 ft | 8011 | 0.8325 | 0.3206 | 19.17 | 0.8835 | 0.813 |

## Strictly comparable subset (LOCO hold-out INTERSECT pooled hold-out, n=1595)

| model | n | type_acc | type_macro_f1 | year_mae | period_acc | floors_mae |
|---|---:|---:|---:|---:|---:|---:|
| DINOv2 frozen | 1595 | 0.8984 | 0.3598 | 18.91 | 0.8853 | 0.573 |
| ResNet-50 ft | 1595 | 0.8382 | 0.3080 | 18.91 | 0.8821 | 0.818 |
| InternVL3 (ZS) | 1595 | 0.4815 | 0.1914 | 30.90 | 0.8276 | 0.668 |
