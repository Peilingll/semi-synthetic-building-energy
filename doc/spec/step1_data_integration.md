# Step 1：資料串接（Data Integration）

## 1. 目標

證明 BAG + 3D BAG + EP-Online 三個公開資料源可以透過 `pand_id` 串接起來，產出一份包含建築基本資訊、3D 幾何屬性、能源標章的整合資料集。

**MVP 研究區域：** Delft（台夫特）

**輸入：** Delft 的 Bounding Box 座標

**輸出：** `data/processed/bag_3dbag_ep_joined.parquet`

**驗收標準：**

- 至少 50 棟建築三個資料源都有資料（inner join 後的行數）
- `Energieklasse` 分布合理（不全是同一個 label，至少涵蓋 3 種以上）
- `bouwjaar` 範圍涵蓋至少 3 個 TABULA period（...1964, 1965-1974, 1975-1991, 1992-2005, 2006-...）

---

## 2. 資料源總覽

| 資料源 | 取得方式 | 格式 | 串接用 ID | 關鍵欄位 |
|--------|----------|------|-----------|----------|
| BAG | PDOK WFS API | GeoJSON (via WFS) | `identificatie` | bouwjaar, geometry, gebruiksdoel |
| 3D BAG | 3D BAG WFS API | GeoJSON (via WFS) | `identificatie` (含前綴) | b_dak_type, h_dak_max, 等 3D 屬性 |
| EP-Online | 本地 CSV 檔案 | CSV (分號分隔) | `BAGPandIDs` | Energieklasse, EnergieIndex, Gebouwtype |

---

## 3. 實作步驟

### 3.1 定義研究區域 (Bounding Box)

Delft 市中心區域的座標範圍：

| 座標系統 | min_x / min_lon | min_y / min_lat | max_x / max_lon | max_y / max_lat |
|----------|-----------------|-----------------|-----------------|-----------------|
| EPSG:28992 (RD New) | 80000 | 447000 | 86000 | 452000 |
| EPSG:4326 (WGS84) | 4.33 | 51.98 | 4.40 | 52.03 |

> **注意：** PDOK WFS 使用 EPSG:28992，3D BAG WFS 也使用 EPSG:28992。實際 bbox 可根據測試需求微調（例如先用更小的範圍測試 API 連線）。

**程式碼範例：**

```python
# config.yaml
study_area:
  name: "Delft"
  bbox_rd:  [80000, 447000, 86000, 452000]   # EPSG:28992 [min_x, min_y, max_x, max_y]
  bbox_wgs: [4.33, 51.98, 4.40, 52.03]       # EPSG:4326 [min_lon, min_lat, max_lon, max_lat]
  crs: "EPSG:28992"
```

---

### 3.2 從 PDOK 抓取 BAG 基本資料

**API 資訊：**

- **類型：** OGC WFS 2.0
- **URL：** `https://service.pdok.nl/lv/bag/wfs/v2_0`
- **圖層：** `bag:pand`
- **座標系統：** EPSG:28992

**取得欄位：**

| 欄位名稱 | 說明 | 用途 |
|----------|------|------|
| `identificatie` | BAG pand ID (16 位數字字串) | 串接用主鍵 |
| `bouwjaar` | 建造年份 | TABULA period 分類 |
| `geometry` | 建築物多邊形 (Polygon) | 空間分析、面積計算 |
| `status` | 建築物狀態 | 過濾：只保留 "Pand in gebruik" |

**WFS 請求參數：**

| 參數 | 值 |
|------|-----|
| `service` | WFS |
| `version` | 2.0.0 |
| `request` | GetFeature |
| `typeName` | bag:pand |
| `bbox` | 80000,447000,86000,452000 |
| `outputFormat` | application/json |
| `srsName` | EPSG:28992 |
| `count` | 10000 (需分頁處理) |

**注意事項：**

- WFS 預設有回傳筆數上限（通常 1000-10000），需使用 `startIndex` 分頁抓取
- 建議使用 `geopandas.read_file()` 搭配 WFS URL，它會自動處理 GeoJSON 解析
- 過濾條件：`status = 'Pand in gebruik'`（排除已拆除或計畫中的建築）

**程式碼範例：**

```python
import geopandas as gpd

def fetch_bag_pand(bbox: list[float], crs: str = "EPSG:28992") -> gpd.GeoDataFrame:
    """Fetch BAG pand data from PDOK WFS for a given bounding box."""
    wfs_url = "https://service.pdok.nl/lv/bag/wfs/v2_0"
    params = {
        "service": "WFS",
        "version": "2.0.0",
        "request": "GetFeature",
        "typeName": "bag:pand",
        "bbox": ",".join(str(c) for c in bbox),
        "outputFormat": "application/json",
        "srsName": crs,
        "count": 10000,
    }

    all_features = []
    start_index = 0

    while True:
        params["startIndex"] = start_index
        gdf = gpd.read_file(wfs_url, params=params)
        if gdf.empty:
            break
        all_features.append(gdf)
        if len(gdf) < params["count"]:
            break
        start_index += params["count"]

    result = gpd.pd.concat(all_features, ignore_index=True)
    result = result[result["status"] == "Pand in gebruik"]
    return result
```

---

### 3.3 從 3D BAG 抓取 3D 屬性資料

**API 資訊：**

- **類型：** OGC WFS
- **URL：** `https://data.3dbag.nl/api/BAG3D/wfs`
- **圖層：** `BAG3D:lod12`（或 `BAG3D:lod22`，視需要的精細度而定）
- **座標系統：** EPSG:28992

**取得欄位（依圖層版本可能不同）：**

| 欄位名稱 | 說明 | 用途 |
|----------|------|------|
| `identificatie` | BAG pand ID（可能含 `NL.IMBAG.Pand.` 前綴） | 串接用主鍵（需清洗） |
| `b_dak_type` | 屋頂類型 | 建築分類特徵 |
| `h_dak_max` | 屋頂最大高度 (m) | 建築高度 |
| `h_dak_min` | 屋頂最小高度 (m) | 建築高度 |
| `h_maaiveld` | 地面高度 (m) | 建築淨高計算 |
| `b_footprint_area` | 建築基底面積 (m²) | 面積特徵 |
| `geometry` | 建築物多邊形 | 空間分析 |

> **注意：** 3D BAG 的 WFS 回傳的 `identificatie` 可能帶有前綴 `NL.IMBAG.Pand.`，需要在子步驟五中清洗。實際可用欄位名稱請先透過 `GetCapabilities` 或小範圍測試請求確認。

**程式碼範例：**

```python
def fetch_3dbag(bbox: list[float], crs: str = "EPSG:28992") -> gpd.GeoDataFrame:
    """Fetch 3D BAG building attributes from WFS for a given bounding box."""
    wfs_url = "https://data.3dbag.nl/api/BAG3D/wfs"
    params = {
        "service": "WFS",
        "version": "2.0.0",
        "request": "GetFeature",
        "typeName": "BAG3D:lod12",
        "bbox": ",".join(str(c) for c in bbox),
        "outputFormat": "application/json",
        "srsName": crs,
        "count": 10000,
    }

    all_features = []
    start_index = 0

    while True:
        params["startIndex"] = start_index
        gdf = gpd.read_file(wfs_url, params=params)
        if gdf.empty:
            break
        all_features.append(gdf)
        if len(gdf) < params["count"]:
            break
        start_index += params["count"]

    return gpd.pd.concat(all_features, ignore_index=True)
```

---

### 3.4 讀取與預處理本地 EP-Online 資料

**檔案資訊：**

| 項目 | 值 |
|------|-----|
| 路徑 | `data/raw/v20260401_v4_csv/v20260401_v4_csv.csv` |
| 格式 | CSV，分號 (`;`) 分隔 |
| 編碼 | UTF-8 |
| 行數 | ~6,122,234 筆 |
| 欄位數 | 42 欄 |
| 檔案大小 | ~1.5 GB |

**需要的欄位：**

| 原始欄位名稱 | 說明 | 對應到最終 schema |
|-------------|------|------------------|
| `BAGPandIDs` | BAG 建築物 ID (16 位數字字串) | 串接用主鍵 |
| `Energieklasse` | 能源標章 (A++, A+, A, B, C, D, E, F, G) | energy_label |
| `EnergieIndex` | 能源指標 (數值) | energy_index |
| `Gebouwtype` | 建築類型 (Rijwoning, Appartement, ...) | building_type 參考 |
| `Gebouwklasse` | 建築等級 (W=住宅, U=公用) | 過濾用 |
| `Bouwjaar` | 建造年份 | 交叉驗證用 |
| `Postcode` | 郵遞區號 | 初步篩選 Delft 區域 |
| `Status` | 狀態 | 過濾用 |
| `Registratiedatum` | 登記日期 | 去重複用（保留最新） |

**Delft 郵遞區號範圍：** `2600-2629`（前四碼 `26xx`）

**載入策略（效能考量）：**

由於 CSV 檔案有 1.5 GB，不建議全部讀入記憶體。建議策略：

1. **方法 A — chunked reading：** 使用 `pd.read_csv(chunksize=...)` 分塊讀取，每塊篩選 Delft 區域後合併
2. **方法 B — column filtering：** 只讀取需要的欄位 (`usecols`)，再篩選 Postcode

**程式碼範例：**

```python
import pandas as pd

def load_ep_online(
    csv_path: str,
    postcode_prefix: str = "26",
    usecols: list[str] | None = None,
) -> pd.DataFrame:
    """Load EP-Online data filtered by postcode prefix (e.g. '26' for Delft)."""
    if usecols is None:
        usecols = [
            "BAGPandIDs",
            "Energieklasse",
            "EnergieIndex",
            "Gebouwtype",
            "Gebouwklasse",
            "Bouwjaar",
            "Postcode",
            "Status",
            "Registratiedatum",
        ]

    chunks = []
    for chunk in pd.read_csv(
        csv_path,
        sep=";",
        usecols=usecols,
        dtype={"BAGPandIDs": str, "Postcode": str},
        chunksize=100_000,
    ):
        # Filter by postcode prefix for Delft area
        mask = chunk["Postcode"].str.startswith(postcode_prefix, na=False)
        filtered = chunk[mask]
        if not filtered.empty:
            chunks.append(filtered)

    df = pd.concat(chunks, ignore_index=True)

    # Keep only residential buildings
    df = df[df["Gebouwklasse"] == "W"]

    # Keep only existing buildings
    df = df[df["Status"] == "Bestaand"]

    # Deduplicate: keep the latest registration per BAGPandIDs
    df["Registratiedatum"] = pd.to_datetime(df["Registratiedatum"])
    df = df.sort_values("Registratiedatum").drop_duplicates(
        subset=["BAGPandIDs"], keep="last"
    )

    return df
```

---

### 3.5 資料清洗與 ID 統一

三個資料源的 pand ID 格式可能不同，需要統一為 **16 位數字字串**。

**各資料源 ID 格式比較：**

| 資料源 | 欄位名稱 | 範例值 | 需要的處理 |
|--------|----------|--------|-----------|
| BAG (PDOK WFS) | `identificatie` | `0503100000012345` | 無（已是 16 位數字字串） |
| 3D BAG WFS | `identificatie` | `NL.IMBAG.Pand.0503100000012345` | 去除 `NL.IMBAG.Pand.` 前綴 |
| EP-Online CSV | `BAGPandIDs` | `0503100000012345` | 無（已是 16 位數字字串） |

> **重要：** EP-Online 的 `BAGPandIDs` 欄位可能包含多個 ID（以逗號或分號分隔），需要 explode 處理。同一棟建築 (pand) 可能有多筆 EP 認證（不同住戶/單元），需要決定聚合策略。

**程式碼範例：**

```python
def clean_pand_id(raw_id: str) -> str:
    """Normalize pand ID to 16-digit string."""
    if pd.isna(raw_id):
        return None
    cleaned = str(raw_id).strip()
    # Remove common prefixes
    prefix = "NL.IMBAG.Pand."
    if cleaned.startswith(prefix):
        cleaned = cleaned[len(prefix):]
    # Ensure 16-digit zero-padded string
    cleaned = cleaned.zfill(16)
    return cleaned


def clean_3dbag_ids(gdf: gpd.GeoDataFrame, id_col: str = "identificatie") -> gpd.GeoDataFrame:
    """Clean 3D BAG identifiers by removing the NL.IMBAG.Pand. prefix."""
    gdf = gdf.copy()
    gdf["pand_id"] = gdf[id_col].apply(clean_pand_id)
    return gdf


def clean_ep_ids(df: pd.DataFrame, id_col: str = "BAGPandIDs") -> pd.DataFrame:
    """Clean EP-Online pand IDs; handle multi-ID entries by exploding."""
    df = df.copy()
    # Split multi-ID entries (some records may list multiple pand IDs)
    df[id_col] = df[id_col].astype(str)
    df = df.assign(**{id_col: df[id_col].str.split(",")}).explode(id_col)
    df["pand_id"] = df[id_col].apply(clean_pand_id)
    df = df.dropna(subset=["pand_id"])
    return df
```

---

### 3.6 執行三方串接 (Join/Merge)

**串接策略：**

```
BAG (PDOK)          3D BAG              EP-Online
┌──────────┐    ┌──────────┐    ┌──────────────────┐
│ pand_id  │    │ pand_id  │    │ pand_id          │
│ bouwjaar │    │ h_dak_max│    │ Energieklasse    │
│ geometry │    │ b_dak_typ│    │ EnergieIndex     │
│ status   │    │ footprint│    │ Gebouwtype       │
└────┬─────┘    └────┬─────┘    └───────┬──────────┘
     │               │                  │
     └───────┬───────┘                  │
             │ inner join               │
             │ on pand_id               │
             ▼                          │
     ┌───────────────┐                  │
     │ bag_3dbag     │                  │
     │ (BAG + 3D)    │                  │
     └───────┬───────┘                  │
             │                          │
             └──────────┬───────────────┘
                        │ inner join
                        │ on pand_id
                        ▼
              ┌───────────────────┐
              │ bag_3dbag_ep      │
              │ (final joined)    │
              └───────────────────┘
```

**Join 類型：** `inner join` — 確保每棟建築都同時擁有三個來源的資訊

**過濾條件（join 後）：**

1. `bouwjaar` 合理範圍：1800 ≤ bouwjaar ≤ 2026
2. `Energieklasse` 不為空
3. 移除重複的 `pand_id`

**程式碼範例：**

```python
def join_all_sources(
    bag_gdf: gpd.GeoDataFrame,
    bag3d_gdf: gpd.GeoDataFrame,
    ep_df: pd.DataFrame,
) -> gpd.GeoDataFrame:
    """Join BAG, 3D BAG, and EP-Online data on pand_id."""
    # Step 1: Ensure all sources have a clean pand_id column
    bag_gdf = bag_gdf.copy()
    bag_gdf["pand_id"] = bag_gdf["identificatie"].apply(clean_pand_id)

    bag3d_gdf = clean_3dbag_ids(bag3d_gdf)
    ep_df = clean_ep_ids(ep_df)

    # Step 2: Join BAG + 3D BAG
    bag_3dbag = bag_gdf.merge(
        bag3d_gdf.drop(columns=["geometry"], errors="ignore"),
        on="pand_id",
        how="inner",
        suffixes=("", "_3dbag"),
    )

    # Step 3: Join with EP-Online
    joined = bag_3dbag.merge(
        ep_df,
        on="pand_id",
        how="inner",
        suffixes=("", "_ep"),
    )

    # Step 4: Post-join filtering
    joined = joined[
        (joined["bouwjaar"] >= 1800)
        & (joined["bouwjaar"] <= 2026)
        & (joined["Energieklasse"].notna())
        & (joined["Energieklasse"] != "")
    ]

    # Step 5: Deduplicate by pand_id
    joined = joined.drop_duplicates(subset=["pand_id"])

    return joined
```

---

### 3.7 驗證與產出

**驗證項目：**

| 檢查項目 | 通過條件 | 說明 |
|----------|---------|------|
| 總筆數 | ≥ 50 | inner join 後至少 50 棟建築 |
| Energieklasse 分布 | ≥ 3 種不同 label | 不全是同一個 label |
| bouwjaar 涵蓋範圍 | ≥ 3 個 TABULA period | 涵蓋不同建築年代 |
| pand_id 唯一性 | 無重複 | 每棟建築只出現一次 |
| geometry 完整性 | 無空值 | 每棟建築都有幾何資訊 |

**TABULA period 對照表：**

| Period 編號 | 年份範圍 | Label |
|-------------|---------|-------|
| 1 | ≤ 1964 | ...1964 |
| 2 | 1965 - 1974 | 1965-1974 |
| 3 | 1975 - 1991 | 1975-1991 |
| 4 | 1992 - 2005 | 1992-2005 |
| 5 | ≥ 2006 | 2006-... |

**產出檔案：**

- 路徑：`data/processed/bag_3dbag_ep_joined.parquet`
- 格式：Apache Parquet
- 預期欄位：

```
pand_id             : str     # 16-digit BAG pand ID
bouwjaar            : int     # construction year (from BAG)
geometry            : geometry# building polygon (from BAG)
h_dak_max           : float   # max roof height (from 3D BAG)
b_dak_type          : str     # roof type (from 3D BAG)
b_footprint_area    : float   # footprint area (from 3D BAG)
Energieklasse       : str     # energy label A++ to G (from EP-Online)
EnergieIndex        : float   # energy index (from EP-Online)
Gebouwtype          : str     # building type (from EP-Online)
```

> **注意：** 實際欄位名稱以 3D BAG WFS 回傳的結果為準，需在首次測試時確認並更新此清單。

**程式碼範例：**

```python
def validate_and_save(gdf: gpd.GeoDataFrame, output_path: str) -> dict:
    """Validate the joined dataset and save to parquet."""
    report = {}

    # Check total count
    report["total_buildings"] = len(gdf)
    assert len(gdf) >= 50, f"Only {len(gdf)} buildings, need at least 50"

    # Check energy label distribution
    label_counts = gdf["Energieklasse"].value_counts()
    report["unique_labels"] = len(label_counts)
    report["label_distribution"] = label_counts.to_dict()
    assert len(label_counts) >= 3, f"Only {len(label_counts)} unique labels"

    # Check TABULA period coverage
    def classify_period(year):
        if year <= 1964:
            return "...1964"
        elif year <= 1974:
            return "1965-1974"
        elif year <= 1991:
            return "1975-1991"
        elif year <= 2005:
            return "1992-2005"
        else:
            return "2006-..."

    gdf["tabula_period"] = gdf["bouwjaar"].apply(classify_period)
    period_counts = gdf["tabula_period"].value_counts()
    report["unique_periods"] = len(period_counts)
    report["period_distribution"] = period_counts.to_dict()
    assert len(period_counts) >= 3, f"Only {len(period_counts)} TABULA periods"

    # Check uniqueness
    assert gdf["pand_id"].is_unique, "Duplicate pand_id found"

    # Save
    gdf.to_parquet(output_path, index=False)
    report["output_path"] = output_path

    return report
```

---

## 4. 程式碼結構

```
src/
├── __init__.py
├── config.py           # load config.yaml
└── data_loader.py      # all functions in this spec

config.yaml             # bbox, paths, parameters
```

### config.yaml 完整範例

```yaml
study_area:
  name: "Delft"
  bbox_rd: [80000, 447000, 86000, 452000]
  bbox_wgs: [4.33, 51.98, 4.40, 52.03]
  crs: "EPSG:28992"

data_paths:
  ep_online_csv: "data/raw/v20260401_v4_csv/v20260401_v4_csv.csv"
  processed_dir: "data/processed"
  joined_output: "data/processed/bag_3dbag_ep_joined.parquet"

ep_online:
  postcode_prefix: "26"
  building_class: "W"

wfs:
  bag_url: "https://service.pdok.nl/lv/bag/wfs/v2_0"
  bag_layer: "bag:pand"
  bag3d_url: "https://data.3dbag.nl/api/BAG3D/wfs"
  bag3d_layer: "BAG3D:lod12"
  page_size: 10000

filters:
  bouwjaar_min: 1800
  bouwjaar_max: 2026
  min_buildings: 50
```

---

## 5. 風險與注意事項

| 風險 | 影響 | 應對策略 |
|------|------|---------|
| WFS 請求逾時或被限流 | 無法抓取完整資料 | 設定 retry + timeout，必要時縮小 bbox |
| 3D BAG WFS 欄位名稱不確定 | 程式碼需調整 | 先用小範圍 `GetCapabilities` 確認 schema |
| EP-Online 多筆認證對同一棟建築 | join 後有重複 | 以 `Registratiedatum` 最新一筆為準 |
| BAGPandIDs 含多個 ID | 一對多展開 | explode 處理後再 join |
| 記憶體不足（6M 行 CSV） | 程式崩潰 | chunked reading + postcode 預篩 |
| 3D BAG WFS 服務不穩定 | 可能暫時無法存取 | 先用本地快取，WFS 抓成功後存為 parquet |

---

## 6. 測試計畫

1. **單元測試：** `clean_pand_id()` 函數對各種輸入格式的處理正確性
2. **整合測試：** 用極小 bbox（如 Delft 某一個街區）跑完全流程
3. **手動驗證：** 抽查 3-5 棟建築，在 [BAG Viewer](https://bagviewer.kadaster.nl/) 和 [EP-Online](https://www.ep-online.nl/) 網站上確認資料一致性
