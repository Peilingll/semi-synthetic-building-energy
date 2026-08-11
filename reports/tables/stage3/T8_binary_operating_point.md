# T8 — Binary task at two operating points (hold-out, n=2,016)

Positive class = `D-G`. Dev base rate P(D-G) = **0.297**; the rate-matched point labels that same fraction of the hold-out (highest scores first). ROC-AUC is threshold-free and shared by both columns.

| route | mF1 @0.5 | bal.acc @0.5 | D-G recall @0.5 | mF1 @rate | bal.acc @rate | D-G recall @rate | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| M0 | 0.4124 | 0.5000 | 0.0000 | 0.4124 | 0.5000 | 0.0000 | 0.5000 | 0.2981 |
| M1 | 0.4919 | 0.5289 | 0.0982 | 0.5756 | 0.5754 | 0.4010 | 0.6466 | 0.4169 |
| M3-DINOv2 | 0.4904 | 0.5223 | 0.1082 | 0.5419 | 0.5417 | 0.3527 | 0.5819 | 0.3535 |
| M3-ResNet50 | 0.4879 | 0.5185 | 0.1098 | 0.5283 | 0.5281 | 0.3311 | 0.5699 | 0.3456 |
| M3-VLMv3 | 0.5276 | 0.5285 | 0.3594 | 0.5024 | 0.5140 | 0.1764 | 0.5346 | 0.3220 |
| M2-DINOv2 | 0.5971 | 0.6003 | 0.4692 | 0.6029 | 0.6028 | 0.4409 | 0.6614 | 0.4311 |
| M2-ResNet50 | 0.5658 | 0.5658 | 0.3910 | 0.5650 | 0.5649 | 0.3877 | 0.6151 | 0.4034 |

Rate-matched thresholds on P(D-G): M0 0.298, M1 0.358, M3-DINOv2 0.377, M3-ResNet50 0.379, M3-VLMv3 0.515, M2-DINOv2 0.516, M2-ResNet50 0.506.

