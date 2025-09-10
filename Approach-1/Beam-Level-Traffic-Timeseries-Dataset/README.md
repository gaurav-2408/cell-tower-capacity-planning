---
license: mit
task_categories:
- time-series-forecasting
task_ids:
- univariate-time-series-forecasting
- multivariate-time-series-forecasting
pretty_name: Beam-Level (5G) Time-Series Dataset
configs:
- config_name: DLPRB
  description: Downlink Physical Resource Block (DLPRB) time-series data.
  data_files:
  - split: train_0w_5w
    path: data/train/DLPRB_train_0w-5w.csv
  - split: test_5w_6w
    path: data/test/DLPRB_test_5w-6w.csv
  - split: test_10w_11w
    path: data/test/DLPRB_test_10w-11w.csv
- config_name: DLThpVol
  description: Downlink Throughput Volume (DLThpVol) time-series data.
  data_files:
  - split: train_0w_5w
    path: data/train/DLThpVol_train_0w-5w.csv
  - split: test_5w_6w
    path: data/test/DLThpVol_test_5w-6w.csv
  - split: test_10w_11w
    path: data/test/DLThpVol_test_10w-11w.csv
- config_name: DLThpTime
  description: Downlink Throughput Time (DLThpTime) time-series data.
  data_files:
  - split: train_0w_5w
    path: data/train/DLThpTime_train_0w-5w.csv
  - split: test_5w_6w
    path: data/test/DLThpTime_test_5w-6w.csv
  - split: test_10w_11w
    path: data/test/DLThpTime_test_10w-11w.csv
- config_name: MR_number
  description: Measurement Report Number (MR_number) time-series data.
  data_files:
  - split: train_0w_5w
    path: data/train/MR_number_train_0w-5w.csv
  - split: test_5w_6w
    path: data/test/MR_number_test_5w-6w.csv
  - split: test_10w_11w
    path: data/test/MR_number_test_10w-11w.csv
language:
- en
tags:
- wireless
---



# 📶 Beam-Level (5G) Time-Series Dataset

This dataset introduces a **novel multivariate time series** specifically curated to support research in enabling **accurate prediction of KPIs** across communication networks, as illustrated below:

<p align="center">
  <img src="images/network.png" alt="Base station, cells, and beams" />
</p>

Precise forecasting of network traffic is critical for optimizing **network management** and enhancing **resource allocation efficiency**. This task is of both **practical and theoretical importance** to researchers in networking and machine learning, offering a strong benchmark for state-of-the-art (SOTA) time series models.

---

## 📂 Dataset Overview

The dataset comprises:
* **2,880 Beams** across 30 Base Stations (3 Cells per Station, 32 Beams per Cell).
* **Duration:** 5 weeks + 2 target weeks, totaling up to 840 training hours and 1176 total hours per beam.

---

## 📁 Available CSV Files

### 🏋️‍♂️ Training Set (Weeks 0–5)

| File Name | Metric |
|---|---|
| `DLThpVol_train_0w-5w.csv` | Downlink throughput volume |
| `DLThpTime_train_0w-5w.csv` | Throughput transmission time |
| `DLPRB_train_0w-5w.csv` | PRB (Physical Resource Block) usage |
| `MR_number_train_0w-5w.csv` | User count (Measurement Reports) |

### 🎯 Forecast Targets

#### 📆 6th Week (Week 5–6)

| File Name | Metric |
|---|---|
| `DLThpVol_test_5w-6w.csv` | Downlink throughput volume |
| `DLThpTime_test_5w-6w.csv` | Throughput transmission time |
| `DLPRB_test_5w-6w.csv` | PRB usage |
| `MR_number_test_5w-6w.csv` | User count |

#### 📆 11th Week (Week 10–11)

| File Name | Metric |
|---|---|
| `DLThpVol_test_10w-11w.csv` | Downlink throughput volume |
| `DLThpTime_test_10w-11w.csv` | Throughput transmission time |
| `DLPRB_test_10w-11w.csv` | PRB usage |
| `MR_number_test_10w-11w.csv` | User count |

---

## 🧪 Dataset Splits

<p align="center">
  <img src="images/dataset_split.png" alt="Dataset train/forecast split" />
</p>

The dataset is split into a **Training Set** (first 5 weeks) and **Forecast Targets** for Week 6 (immediate future) and Week 11 (long-term future).

---

## 📄 Data Format

Each CSV file contains a `Time` column and multiple beam columns (e.g., `0_0_0` to `29_2_31`). The `Time` column ranges from `0–839` for training (weeks 1–6), `0–167` for week 6, and `168–335` for week 11. Each beam column uniquely identifies one of the **2,880 beams** across 30 base stations.

---

## 📚 Citation

If you use this dataset in your research, please cite:

> **L. Fechete et al.**, *Goal-Oriented Time-Series Forecasting: Foundation Framework Design*, arXiv:2504.17493 (2025)

---

## 🔗 Code Repository

The official codebase for working with this dataset is available here: 👉 [https://github.com/netop-team/gotsf](https://github.com/netop-team/gotsf)
