# Table 2d — Intra-pand EPC label entropy audit (manifest pands)

Labels normalised A+..A++++ -> A. modal share = fraction of a pand's
certificates carrying its most common label. Oracle ceiling = accuracy of
predicting each pand's modal label scored against a random unit's label.

## All certificates (any year)

- pands with >=1 certificate: **10093**
- pands with >=2 certificates: **5895** (58.4%)
- certificates per pand (all): median 2, mean 3.2, max 235
- among multi-cert pands, >=2 distinct labels: **4539** (77.0% of multi)

| quantity | all pands | multi-cert pands only |
|---|---:|---:|
| mean modal share (oracle acc ceiling) | **0.7918** | 0.6435 |
| median modal share | 1.0000 | 0.5000 |
| p25 modal share | 0.5000 | 0.5000 |
| mean entropy (bits) | 0.5237 | 0.8966 |
| P(latest == modal) | 0.8803 | 0.7951 |
| median intra-pand PF std (kWh/m2.yr) | — | 35.8 |

### by building type

| type | pands | % multi-cert | median n_certs | mean modal share | mean entropy |
|---|---:|---:|---:|---:|---:|
| AB | 8908 | 65.0% | 2 | 0.7670 | 0.5856 |
| TH | 990 | 4.8% | 1 | 0.9876 | 0.0301 |
| SFH | 131 | 0.0% | 1 | 1.0000 | 0.0000 |
| MFH | 46 | 95.7% | 6 | 0.7535 | 0.7112 |

## NTA 8800 era only (reg >= 2021, matches pipeline filter)

- pands with >=1 certificate: **10093**
- pands with >=2 certificates: **5893** (58.4%)
- certificates per pand (all): median 2, mean 3.2, max 235
- among multi-cert pands, >=2 distinct labels: **4537** (77.0% of multi)

| quantity | all pands | multi-cert pands only |
|---|---:|---:|
| mean modal share (oracle acc ceiling) | **0.7919** | 0.6436 |
| median modal share | 1.0000 | 0.5000 |
| p25 modal share | 0.5000 | 0.5000 |
| mean entropy (bits) | 0.5235 | 0.8965 |
| P(latest == modal) | 0.8803 | 0.7950 |
| median intra-pand PF std (kWh/m2.yr) | — | 35.8 |

### by building type

| type | pands | % multi-cert | median n_certs | mean modal share | mean entropy |
|---|---:|---:|---:|---:|---:|
| AB | 8908 | 65.0% | 2 | 0.7671 | 0.5854 |
| TH | 990 | 4.8% | 1 | 0.9876 | 0.0301 |
| SFH | 131 | 0.0% | 1 | 1.0000 | 0.0000 |
| MFH | 46 | 95.7% | 6 | 0.7535 | 0.7112 |

