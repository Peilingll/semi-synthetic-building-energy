# Full Manuscript Terminology Audit

## Implementation status, 2026-09-06

The recommendations in this audit have now been applied to the thesis source. The certificate consistency analysis was rerun after restricting the input to the 10,086-building experimental dataset. Of these buildings, 10,075 could be linked back to at least one raw residential certificate, and all thesis values now use that result. Chapter 3 defines the eligible reference set, experimental dataset, development set, fixed holdout set, comparison subset, reference building attributes, and predicted building attributes. Reader-facing terminology for vision configurations, energy class prediction conditions, and the floor-area-weighted relative difference has also been standardised across the manuscript and appendices.

## 1. Audit scope and principal conclusion

This audit reviews the terminology used in the thesis source files from the Abstract through the Appendices. It focuses on the three issues requested for review:

1. the names of the complete dataset, data partitions, holdout set, and analysis subsets;
2. the terms used for building type, construction year, and floor count;
3. the nouns and denominators used when reporting numbers of buildings.

The revised manuscript no longer uses `common dataset` or `shared dataset` to describe the evaluation data. It distinguishes the fixed holdout set of 2,018 buildings from the analysis specific subsets of 2,016 or 2,014 buildings that require valid model outputs or geometry.

The revised manuscript uses **building attributes** as the canonical umbrella term for **building type, construction year, and floor count**. Source and prediction status are expressed with the modifiers **reference** and **predicted**.

The revised manuscript uses `dataset` for the complete linked data product, `set` for a partition, `subset` for an analysis specific selection, and `buildings` when reporting the denominator. Unless otherwise stated, a building count means the number of unique BAG building identifiers, `pand_id`.

## 2. Direct answers to the three questions

### 2.1 How should the holdout data be described?

**Recommended canonical term:** `fixed holdout set`

`Fixed` is accurate because the building identifiers in this set were selected before model evaluation, stored separately, and excluded from model fitting and hyperparameter selection. The repository also stores a checksum for this set. The definition is supported by the split implementation in `src/stage1/splits.py` and is stated in `content/03_method.tex`, lines 189 to 201.

Use the terminology as follows:

| Context | Canonical wording | Meaning |
|---|---|---|
| First definition | `fixed holdout set` | The predefined partition containing 2,018 unique BAG buildings |
| Later references | `holdout set` | Short form referring to the same predefined partition |
| Reporting its denominator | `the 2,018 buildings in the holdout set` | Explicit count and unit |
| Experiment III complete case comparison | `comparison subset` | The 2,016 holdout buildings with valid outputs for every evaluated energy class prediction condition |
| Experiment II physical comparison | `thermal evaluation subset` followed by the exact denominator | The buildings with valid predicted attributes and reconstructed areas for the named configuration |

Do not use the following expressions as synonyms:

| Avoid | Reason |
|---|---|
| `holdout dataset` | The holdout is a partition of the experimental dataset, not a separate dataset |
| `fixed holdout` as a noun | The omitted noun `set` makes the referent less explicit |
| `common dataset` | It does not identify whether the full experimental dataset or a complete case subset is meant |
| `shared dataset` | It suggests joint ownership or access rather than identical evaluated building identifiers |
| `common holdout` or `shared holdout` | Neither term states the inclusion criterion or denominator |
| `same holdout buildings` without a number | This is inaccurate when valid outputs differ by configuration |

The following forms were recorded before implementation and guided the revision:

| Current form | Number of source locations | Assessment |
|---|---:|---|
| `fixed holdout set` | 3 | Correct at first definition and when the fixed status matters |
| `fixed holdout` | 9 | Understandable but should become `fixed holdout set` or `holdout set` |
| `holdout set` | 2 | Correct after the full term has been defined |
| `holdout buildings` | 6 | Acceptable only when the count or inclusion rule is clear |
| `comparison subset` | 5 | Correct for the 2,016 building complete case comparison in Experiment III |
| `same buildings` | 11 | Appropriate as a comparison statement only when the actual building identifiers are identical |
| `common dataset` | 0 | Correctly absent |
| `shared dataset` | 0 | Correctly absent |

Two claims identified during the audit were corrected because they exceeded the actual denominator evidence:

1. Chapter 3 now distinguishes the predefined holdout set from the 2,016-building comparison subset used when every energy class prediction condition must have a valid output.
2. Chapter 5 now states that the supervised configurations use all 2,018 holdout buildings for attribute prediction, while InternVL3-2B has 2,016 valid records. Claims about Experiment III refer specifically to the 2,016-building comparison subset.

### 2.2 What term should describe type, year, and floor count?

**Recommended canonical umbrella term:** `building attributes`

Define it once in the Method as follows:

> In this study, building attributes refer to building type, construction year, and floor count.

After this definition, use the following controlled vocabulary:

| Canonical term | Definition | Permitted short form |
|---|---|---|
| `building attributes` | Building type, construction year, and floor count considered together | `attributes` when the three variables are already explicit in the paragraph |
| `reference building attributes` | The values used as references for evaluation: mapped EP-Online building type, BAG construction year, and 3DBAG floor count | `reference attributes` |
| `predicted building attributes` | Building type, construction year, and floor count predicted from SVI | `predicted attributes` |
| `building attribute prediction` | The Stage 2 task that produces the three predicted attributes | `attribute prediction` |
| `building attribute error` | A difference between a predicted attribute and its corresponding reference attribute | `attribute error` |
| `building attribute enrichment` | The broader workflow in which missing building records are supplemented with predicted attributes | `attribute enrichment` |

The phrase `administrative reference attributes` should not be used for all three attributes in this thesis. Building type and construction year originate from administrative sources, but the floor count is obtained from reconstructed 3DBAG data and may use a height based fallback. The accurate umbrella term is therefore `reference building attributes`, followed by source specific wording where provenance matters.

The following variants should be limited or removed from claims about the present study:

| Current or possible variant | Recommendation |
|---|---|
| `semantic building metadata` | Replace with `building attributes` when it refers to type, year, and floor count |
| `semantic attributes` | Replace with the named attributes or `building attributes`; it is currently broader than the evaluated variables |
| `explicit building attributes` | Usually shorten to `building attributes`; use `intermediate building attributes` only when contrasting the attribute based and direct image procedures |
| `structured attributes` | Do not use as a synonym for the three attributes because the structured model also receives U-values and study area |
| `structured inputs` | Reserve for the eight LightGBM inputs: three building attributes, four U-values, and study area |
| `features` | Reserve for model input columns, geometry variables, or learned image feature vectors |
| `parameters` | Reserve for trained model parameters and TABULA thermal parameters; never use for type, year, or floor count |
| `metadata` | Reserve for technical metadata such as image identifiers, dates, and camera information |
| `attribute extraction` | Replace with `building attribute prediction` in the Abstract, Results, and Conclusions |
| `attribute inference` | Use only for the computational act of running InternVL3; name the study task `building attribute prediction without parameter updating` |
| `estimated attributes` | Acceptable when accurately describing terminology used by earlier studies; use `predicted attributes` for this study |
| `inferred attributes` | Acceptable in broad literature discussion; use `predicted attributes` for this study |
| `image derived attributes` or `SVI derived attributes` | Prefer `building attributes predicted from SVI` in principal claims because it states both source and operation |
| `visual attribute extraction` | Replace with `building attribute prediction from SVI` |

### 2.3 How should the numbers of buildings be described?

**Recommended counting rule:** Unless another unit is explicitly named, `n` denotes the number of unique BAG buildings identified by `pand_id`.

The nouns should be chosen by analytical role:

| Use | When to use it |
|---|---|
| `building` | Unit of linkage, partitioning, prediction, bootstrap resampling, and evaluation |
| `image` | One selected facade crop in the SVI manifest |
| `certificate record` | One raw EP-Online record, which may represent a dwelling or registration date |
| `dataset` | The complete linked data product containing building records and associated image records |
| `set` | A defined partition or subset of buildings |
| `subset` | Buildings selected from a larger set by an explicit availability or validity rule |
| `sample` | Use sparingly for statistical sampling statements or comparison with a wider building stock; do not use it as a general synonym for dataset and set |
| `population` | Avoid for the analysed buildings because the image linked data are not a random or complete population of the Dutch building stock |

The manuscript currently contains no use of `population` in the thesis source. This is appropriate. Use `residential building stock` for the wider real world stock and `experimental dataset` or `evaluated buildings` for the buildings analysed in this study.

## 3. Authoritative dataset and denominator hierarchy

The following hierarchy is supported by the current data artefacts, source code, and audit reports.

| Level | Canonical name | Building count | Image count | Definition and permitted use |
|---|---|---:|---:|---|
| 1 | `eligible reference set` | 124,784 | Not applicable | Unique BAG buildings in `stage1_gt.parquet` with an EPC record and valid TABULA-NL match, without requiring SVI. Use this set only to compare composition with the image linked experimental dataset. |
| 2 | `SVI manifest` | 10,104 | 47,238 | Unique BAG buildings with at least one accepted Mapillary facade crop in `svi_manifest.parquet`. This is the image data product before intersection with the reference set. |
| 3 | `experimental dataset` | 10,086 | 47,150 | Intersection of the SVI manifest and eligible reference set after the required geometry and TABULA-NL inputs are available. This is the authoritative dataset for model development and evaluation. |
| 4a | `development set` | 8,068 | 37,822 | The part of the experimental dataset used for training, validation, prompt development, and feature analysis. |
| 4b | `fixed holdout set` | 2,018 | 9,328 | The predefined part of the experimental dataset excluded from model fitting and hyperparameter selection. |
| 5 | `comparison subset` | 2,016 | Do not report unless needed | The strict intersection of holdout buildings with valid outputs for all Experiment III prediction conditions. |
| 6a | `thermal evaluation subset for ResNet-50 and DINOv2` | 2,016 | Not applicable | Holdout buildings with valid predicted attributes and reconstructed areas for these configurations. |
| 6b | `thermal evaluation subset for InternVL3-2B` | 2,014 | Not applicable | Holdout buildings with a valid InternVL3 attribute record and reconstructed areas. |

The count hierarchy should be presented as:

`eligible reference set` and `SVI manifest` → `experimental dataset` → `development set` plus `fixed holdout set` → analysis specific subsets.

This hierarchy resolves the apparent inconsistency among 124,784, 10,104, 10,086, 8,068, 2,018, 2,016, and 2,014. These values do not describe competing versions of one dataset. They describe different stages or subsets and must always be paired with the canonical name and unit.

### 3.1 Certificate consistency analysis after denominator reconciliation

The certificate consistency analysis was recomputed using the 10,086 buildings in the experimental dataset. Of these buildings, 10,075 could be linked back to at least one residential certificate in the raw EP-Online register. The thesis reports both numbers explicitly and calculates certificate consistency statistics only for the 10,075 linked buildings. The earlier 10,093-building result is retained only in the internal count reconciliation report as a historical result and is not used in the thesis.

### 3.2 Study areas rather than an unqualified four cities claim

The manuscript should use `four Dutch study areas` as the canonical geographic term. The data are labelled Amsterdam, Rotterdam, Utrecht, and Delft, but the count audit identifies three building identifiers from Midden-Delfland in the Delft labelled SVI data. If those records remain, `four Dutch cities` is slightly stronger than the underlying identifier geography supports. Define the four study area labels once and use `study area` as the categorical variable in models and tables.

## 4. Controlled terminology for records and data provenance

| Canonical term | Definition | Important boundary |
|---|---|---|
| `BAG building` | A building identified by one normalised 16 digit BAG `pand_id` | This is the unit of linkage and the principal unit of analysis |
| `building reference record` | One final record per `pand_id` containing the three reference building attributes, reconstructed geometry, and the selected registered energy class | It is not a raw BAG or EP-Online row |
| `street view image record` | One accepted facade crop linked to a `pand_id` | Several image records may correspond to one building reference record |
| `certificate record` | One EP-Online registration linked to a dwelling or building identifier and a registration date | Several certificate records may link to one BAG building |
| `reference data` | BAG and EP-Online administrative records together with reconstructed 3DBAG geometry | Not all reference data are administrative or measured |
| `administrative records` | BAG and EP-Online records | Does not include 3DBAG reconstructed geometry |
| `reconstructed geometry` | Geometry derived by 3DBAG from BAG and elevation data | Must not be called surveyed or measured geometry |
| `registered energy class` | The selected EP-Online class used as the prediction target | It is not produced by TABULA-NL |
| `latest registered energy class` | The class from the most recent linked EP-Online record selected for a BAG building | It may represent one dwelling unit or recorded state rather than the complete building |

The term `ground truth` should remain absent from principal claims. The BAG, EP-Online, and 3DBAG values serve as reference values for evaluation, but the thesis identifies source and unit limitations that prevent them from being treated as error free ground truth.

## 5. Controlled terminology for the attribute and archetype chain

The following distinctions should remain explicit throughout Chapters 1 to 5.

| Term | Definition | Must not be conflated with |
|---|---|---|
| `EP-Online residential category` | One of the selected Dutch source categories before operational mapping | The four class prediction target |
| `building type` | The four class target used in this study: SFH, TH, MFH, or AB | The original six EP-Online categories |
| `TABULA-NL size class` | The size class coordinate used in the archetype lookup | A general building type taxonomy |
| `construction year` | The reference or predicted calendar year | Building age or construction period |
| `construction period` | One of six TABULA-NL periods deterministically assigned from construction year | A directly predicted attribute in the main experiment |
| `floor count` | The reference or predicted number of floors used as the third Stage 2 attribute and a LightGBM input | A determinant of the TABULA-NL cell |
| `reference floor count` | The 3DBAG floor count or its documented height based fallback used as the prediction reference | A surveyed measurement |
| `geometric floor count` | The separate derived variable used to calculate total floor area when needed | The Stage 2 floor count reference |
| `archetype cell` | The ordered pair of TABULA-NL size class and construction period | A complete physical model of an individual building |
| `reference archetype cell` | The cell assigned from reference building type and reference construction year | A measured archetype |
| `predicted archetype cell` | The cell assigned from predicted building type and predicted construction year | A cell predicted directly by the vision model |
| `representative U-values` | TABULA-NL wall, roof, floor, and window U-values assigned through a cell lookup | Measured envelope properties |
| `thermal parameters` | When used, should refer specifically to the four representative U-values | The calculated coefficient `h_tr` |
| `floor area normalised transmission heat transfer coefficient`, `h_tr` | A calculated coefficient based on representative U-values and reconstructed areas | Heat demand, energy use, or measured transmission heat loss |

The revision separates the prediction target from the lookup coordinate. The model predicts a four-class building type, and these labels supply the corresponding TABULA-NL residential size class used during archetype assignment.

## 6. Controlled terminology for experiments and models

The current manuscript has removed the earlier M0, M1, M2, and M3 route notation from the thesis source. This is an improvement. The remaining terms can be assigned one function each:

| Canonical term | Use |
|---|---|
| `stage` | One of the four Method stages: dataset construction, building attribute prediction, archetype assignment, or energy class prediction |
| `experiment` | One of the three evaluations aligned with the research objectives |
| `vision configuration` | One complete ResNet-50, DINOv2, or InternVL3 setup, including architecture, parameter updating, image limit, and aggregation |
| `prediction procedure` | Attribute based energy class prediction or direct image energy class prediction |
| `energy class prediction condition` | Reference attribute condition, predicted attribute condition, or direct image condition in Experiment III |
| `model` | A fitted or pretrained computational model within a configuration |
| `information source` | The data supplied to a prediction procedure, such as reference attributes, predicted attributes, or SVI |

`Route` should remain absent. `Configuration` should not refer to the reference attribute and predicted attribute conditions, and `condition` should not be used as a synonym for a ResNet-50 or DINOv2 configuration.

The phrase `information source condition` is understandable, but `energy class prediction condition` is more accurate because the direct image condition changes both the input source and the prediction procedure. The thesis already acknowledges that the comparison does not isolate information source from model family. The table and prose should preserve that limitation.

## 7. Energy target and metric terminology

| Canonical term | Definition and use |
|---|---|
| `Energy Performance Certificate`, `EPC` | The certificate or register record; do not use `EPC` as a synonym for its class |
| `registered energy class` | The categorical target selected from EP-Online |
| `binary energy class task` | Prediction of A to C versus D to G |
| `seven class energy class task` | Prediction of A through G after ratings above A are combined with A |
| `energy class prediction` | Canonical task name; avoid `predicting EPC` |
| `reference attribute condition` | LightGBM supplied with reference building attributes and their assigned U-values |
| `predicted attribute condition` | The same fitted LightGBM model supplied with predicted building attributes and U-values assigned from predicted type and year |
| `direct image condition` | Energy class prediction from SVI without intermediate building attribute outputs |

The prompts in Appendix D may retain the Dutch phrase `energy label` because they are reproduced verbatim as experimental instruments. The surrounding thesis prose should use `registered energy class`.

The earlier term `stock level bias` was replaced with `floor-area-weighted relative difference`. The calculation compares a predicted cell calculation with a reference cell calculation, but neither is validated against measured heat loss. A positive value means that the predicted cell calculation produces a larger aggregate transmission heat transfer coefficient. The Abstract was revised accordingly because the evaluated quantity is a coefficient and the reference is an archetype based calculation rather than measured heat loss.

## 8. Dataset, sample, and set implementation record

The following table records the issues identified before implementation. All actions marked `Adopt` have been applied to the thesis source.

| Location | Current wording | Recommended wording or action | Severity | Decision |
|---|---|---|---|---|
| `00c_abstract.tex:3` | `integrated evaluation dataset` | Use `experimental dataset` if this is the canonical name adopted in Method | Major | Adopt |
| `00c_abstract.tex:3` | `semantic building metadata` | Use `building attributes` and name the three attributes immediately | Major | Adopt |
| `00c_abstract.tex:3` | `attribute extraction` | Use `building attribute prediction` | Major | Adopt |
| `00c_abstract.tex:5` | `overestimated stock-level transmission heat loss` | Report the positive floor area weighted relative difference in `h_tr` from the reference cell calculation | Critical | Adopt |
| `01_introduction.tex:36,43,52` | `administrative reference attributes` | Use `reference building attributes`; specify administrative or reconstructed source only when needed | Critical | Adopt |
| `01_introduction.tex:55` | `image-derived building attributes` | Use `building attributes predicted from SVI` | Major | Adopt |
| `03_method.tex:7` | `building information obtained from street view imagery` | Use `building attributes predicted from SVI` because the evaluated outputs are explicit | Major | Adopt |
| `03_method.tex:40,150,151,181` | Several forms of `linked dataset` | Define the final output once as the `experimental dataset` | Major | Adopt |
| `03_method.tex:193` | `fixed holdout set` | Retain as the authoritative first definition | None | Retain |
| `03_method.tex:201` | `different building samples` | Use `different building partitions`; later report analysis specific subsets separately | Major | Adopt |
| `03_method.tex:207` | Building type equated with TABULA-NL size classes | Define the four class building type target and its use as the lookup size class separately | Major | Adopt |
| `03_method.tex:278` | `Vision Language Model Attribute Inference` | Consider `Building Attribute Prediction with InternVL3-2B` and explain `without parameter updating` in the paragraph | Minor | Discuss |
| `03_method.tex:410` | `structured inputs` | Retain and define the complete eight input vector | None | Retain |
| `03_method.tex:412` | Same partitions imply unchanged evaluation sample | State the predefined partitions first, then define complete case comparison subsets | Critical | Adopt |
| `04_experiments.tex:4,9,26` | `building sample`, `sample and target conditions` | Use `experimental dataset`, `building partitions`, or `evaluated buildings`, depending on referent | Major | Adopt |
| `04_experiments.tex:37` | `curated SVI delivery` | Use `SVI manifest` to match the actual data artefact | Major | Adopt |
| `04_experiments.tex:37` | `fixed holdout` | Use `fixed holdout set` at first mention in the chapter, then `holdout set` | Minor | Adopt |
| `04_experiments.tex:39` | Denominator paragraph | Retain the content but name the 2,016 Experiment III set `comparison subset` at its first occurrence | Major | Adopt |
| `04_experiments.tex:46` | `explicit attributes` and `same building record` | Use `building attributes`; state that outputs are indexed by `pand_id` and report valid denominators | Major | Adopt |
| `04_experiments.tex:184,221,226` | `final evaluation sample`, `complete evaluation sample`, `experimental sample`, `SVI covered sample` | Use `experimental dataset` for 10,086 and `evaluated buildings` when referring to its members | Critical | Adopt |
| `04_experiments.tex:221` | `eligible reference dataset` | Use `eligible reference set` or `124,784 eligible reference buildings` | Major | Adopt |
| `04_experiments.tex:230` | 10,093 building certificate analysis | Recompute using the 10,086-building experimental dataset | Critical | Completed; 10,075 buildings linked to raw records |
| `04_experiments.tex:265,285,331,351,389` | Repeated `fixed holdout` | Use `holdout set` after the definition and include configuration specific denominators in notes | Minor | Adopt |
| `04_experiments.tex:400` | `comparison subset of 2,016 holdout buildings` | Retain; add the complete case inclusion rule at first definition | None | Retain |
| `04_experiments.tex:500,525` | `structured inputs` | Retain only for the eight LightGBM inputs | None | Retain |
| `04_experiments.tex:525` | `visual attribute extraction` | Use `building attribute prediction from SVI` | Major | Adopt |
| `04_experiments.tex:532` | Repeated forms of `evaluation sample`, `sample`, and `target sample` | Use `experimental dataset`, `evaluated buildings`, and `evaluation set` | Major | Adopt |
| `05_conclusions.tex:5,49` | `administrative reference attributes` | Use `reference building attributes` | Critical | Adopt |
| `05_conclusions.tex:29` | `visual attribute extraction` | Use `building attribute prediction from SVI` | Major | Adopt |
| `05_conclusions.tex:33` | `visible or semantic attributes` | Name `building type, construction year, and floor count`, or use `building attributes` | Major | Adopt |
| `05_conclusions.tex:34` | `structured attributes` | Use `structured inputs` or `building attributes and assigned U-values`, according to the intended referent | Major | Adopt |
| `05_conclusions.tex:36` | `explicit building attributes` | Use `building attributes` | Minor | Adopt |
| `05_conclusions.tex:45,46` | `same holdout buildings` | Distinguish the 2,018 holdout set from the 2,016 valid InternVL3 predictions | Critical | Adopt |
| `05_conclusions.tex:54` | `same buildings and data partitions` | Name the comparison subset if the claim concerns Experiment III | Major | Adopt |
| `zb_appendix_b.tex:78` | `structured building attributes` | Use `structured inputs` because city and U-values are also included | Major | Adopt |
| `zc_appendix_c.tex:47` | `2,016 holdout buildings` | Use `2,016 buildings in the comparison subset` | Minor | Adopt |

## 9. Style and compound term consistency

The user preference to avoid unnecessary hyphens can be followed without removing grammatically required hyphens. When two words jointly modify a following noun, the compound should be hyphenated. When the phrase follows the noun, no hyphen is needed.

| Attributive form | Predicative or noun form |
|---|---|
| `building-level evaluation` | `evaluated at building level` |
| `image-level prediction` | `prediction at image level` |
| `class-level recall` | `recall for each class` |
| `stock-level comparison` | `comparison across the building stock` |
| `floor-area-normalised coefficient` | `coefficient normalised by floor area` |
| `image-based procedure` | `procedure based on images` |

The revised manuscript uses the correctly hyphenated attributive form and leaves the unhyphenated phrase only where it follows a noun or preposition.

## 10. Definition placement

The following terms should be defined at their first authoritative use and then reused without redefinition:

| Term | Authoritative definition location |
|---|---|
| `building attributes` | Chapter 3 Method Overview or Stage 2 opening |
| `reference building attributes` | Chapter 3 Data Provenance |
| `predicted building attributes` | Chapter 3 Stage 2 |
| `experimental dataset` | Chapter 3 Final Dataset Integration and Eligibility |
| `development set` and `fixed holdout set` | Chapter 3 Data Partitioning and Cross Validation |
| `comparison subset` | Chapter 4 Evaluation Setup, before Experiment I metrics |
| `building type` and `TABULA-NL size class` | Chapter 3 Stage 2 and Stage 3 mapping paragraphs |
| `registered energy class` | Chapter 3 Data Sources and Stage 4 target definition |
| `vision configuration` | Chapter 3 Vision Configurations |
| `energy class prediction condition` | Chapter 4 Experiment III design |
| `h_tr` | Chapter 4 Experiment II design, with its unit and distinction from heat demand |

The Abstract and Conclusion should use the canonical terms but need not reproduce full operational definitions. Chapter 2 may retain source specific terminology when describing prior studies, provided the synthesis returns to the canonical terms used by this thesis.

## 11. Abbreviation and notation audit

The main abbreviations for UBEM, TABULA, BAG, EPC, SVI, OSM, VLM, CNN, ViT, MLP, LightGBM, LoD, and AoV are present in the abbreviation definitions. TABULA-NL is correctly treated as the name used for the Dutch TABULA typology rather than as a new acronym.

The following items were checked during implementation:

1. `NTA 8800` is defined at first use as the Dutch method for determining building energy performance.
2. `WWR` is defined locally as window to wall ratio and also appears in the symbol definitions. This is sufficient if the List of Symbols remains disabled, but the first textual definition must remain.
3. The symbol file and table captions use lowercase `n` for the number of evaluated buildings.
4. `EPC band`, `EPC grade`, and `energy label` appear mainly when describing external literature or verbatim prompts. The present study should continue to use `registered energy class` in its own methods, results, and conclusions.

## 12. Mandatory integrity audit

### 12.1 Narrative continuity

The revised information chain is clear: SVI predicts building attributes; building type and construction year determine the archetype cell; the cell assigns representative U-values; building attributes and U-values support one energy class prediction procedure; imagery supports a second procedure. The controlled vocabulary assigns one stable noun to each stage.

### 12.2 Cross-section contradiction

The denominator contradiction has been resolved by stating the fixed partition separately from each valid analysis subset. The revised text reports 2,018, 2,016, or 2,014 buildings according to model output and geometry availability.

The provenance contradiction has been resolved by using `reference building attributes` for the combined EP-Online, BAG, and reconstructed 3DBAG values and naming the individual sources where relevant.

### 12.3 Terminology integrity

The canonical terms should be:

`building attributes` → `reference building attributes` or `predicted building attributes` → `reference archetype cell` or `predicted archetype cell` → `representative U-values` → `energy class prediction condition`.

For data:

`SVI manifest` and `eligible reference set` → `experimental dataset` → `development set` plus `fixed holdout set` → `comparison subset` or a named analysis subset.

### 12.4 New-term necessity

No new framework label is needed. In particular, do not reintroduce `route`, `route family`, `common dataset`, `shared dataset`, or `building population`. The proposed terms name existing data objects and analytical roles rather than adding a new conceptual taxonomy.

### 12.5 Content necessity

The definitions of the dataset hierarchy, building attributes, record units, and denominators are necessary because they determine what is compared and which buildings contribute to each metric. Repeated restatements of `fixed`, generic statements about `building information`, and multiple names for the 10,086 building dataset can be removed after the authoritative definitions are established.

### 12.6 Citation function

This terminology revision does not require new citations for study specific names such as `experimental dataset`, `fixed holdout set`, or `comparison subset`. Existing citations should remain attached to source definitions and inherited methods, including BAG, EP-Online, 3DBAG, TABULA-NL, model architectures, and metrics derived from external calculation methods. Terminology used to describe the present study should not be supported by unrelated literature merely to make it sound established.

## 13. Completed implementation

The certificate analysis was recomputed, the canonical definitions were inserted in Chapter 3, and the data hierarchy and analysis denominators were standardised. The revision also separates building type from TABULA-NL size class, separates construction year from construction period, corrects the terminology for $h_{tr}$ and the floor-area-weighted relative difference, and assigns distinct meanings to `vision configuration`, `prediction procedure`, and `energy class prediction condition`. A final acronym, compound modifier, and denominator check was completed after compilation.

## 14. Final recommendation

The thesis should not claim that every evaluation uses one identical set of 2,018 buildings. It should claim that all models use the same **predefined building partition**, while each result reports the **actual number of buildings with valid inputs and outputs**. Experiment III can make the strongest same building comparison because it explicitly uses the 2,016 building comparison subset.

The thesis should use **building attributes** as the sole umbrella term for building type, construction year, and floor count. Their status should be expressed as **reference** or **predicted**, and their sources should be stated separately. Numbers should be reported as counts of **unique BAG buildings**, with `dataset` reserved for the linked data product and `set` or `subset` reserved for partitions and complete case analyses.
