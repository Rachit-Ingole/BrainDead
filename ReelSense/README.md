# 🎬 ReelSense

## Explainable Movie Recommender System with Diversity Optimization

> **BrainDead @ Revelation 2K26 – Problem Statement 1**
> Department of Computer Science and Technology, IIEST Shibpur

---

## 📌 Problem Statement Alignment (BrainDead – PS1)

ReelSense directly addresses **Problem Statement 1** of the BrainDead competition. The objective is to design a **Top‑K Movie Recommendation System** that:

* Generates **personalized recommendations** using hybrid approaches
* Ensures **diversity and catalog coverage** to mitigate popularity bias
* Provides **natural‑language explanations** for each recommendation
* Reports **ranking, diversity, and novelty metrics**

This repository contains the **complete pipeline**, from data preprocessing and EDA to model training, evaluation, and explainability.

---

## 📂 Dataset Description

**Dataset:** MovieLens Latest Small
**Source:** GroupLens Research
**License:** MovieLens Terms of Use

**Dataset Size**

* 100,836 ratings
* 610 users
* 9,742 movies

**Files Used**

* `ratings.csv` – User ratings (0.5 to 5.0)
* `movies.csv` – Movie metadata (title, genres)
* `tags.csv` – User‑assigned free‑form tags
* `links.csv` – External identifiers (IMDB, TMDb)

---

## 🧹 Preprocessing Pipeline

The following preprocessing steps were applied:

* Time‑based **train–test split** (leave‑last‑N ratings per user)
* Removal of noisy and duplicate tags
* Standardization of genre and tag tokens
* Construction of **user–item interaction matrix**
* Timestamp parsing and conversion to datetime
* Normalization of popularity statistics for diversity metrics

---

## 🔍 Exploratory Data Analysis (EDA)

Key analyses performed:

* Rating distribution analysis
* Genre popularity vs. average rating
* User activity histogram (ratings per user)
* Long‑tail distribution of movies
* Temporal trends in rating behavior

EDA visualizations are included in the `/notebooks` directory and are clearly labeled for interpretation.

---

## 🧠 Models Implemented

### 1️⃣ Popularity‑Based Baseline

* Top‑N most‑rated movies
* Top‑N highest‑average‑rated movies

### 2️⃣ Collaborative Filtering

* User–User Collaborative Filtering
* Item–Item Collaborative Filtering

### 3️⃣ Matrix Factorization

* Singular Value Decomposition (SVD)
* Implemented using the **Surprise** library
* Optimized for RMSE minimization

### 4️⃣ Hybrid Recommendation Model

A weighted hybrid of:

* Collaborative scores (SVD)
* Content similarity (genres + tags)
* Bayesian‑smoothed global ratings

---

## 🧠 Hybrid Scoring Logic

The final recommendation score is computed as:

$$
Score_{final} = (\alpha \cdot P_{SVD}) + (\beta \cdot P_{Content}) - (\gamma \cdot Popularity_{norm})
$$

Where:

* $P_{SVD}$ captures latent user preferences
* $P_{Content}$ captures semantic similarity
* $Popularity_{norm}$ penalizes over‑popular items

This formulation improves **novelty and catalog coverage** while retaining accuracy.

---

## ✨ Explainability Layer

Each recommended movie is accompanied by a **human‑readable explanation**, generated using the dominant contributing signal.

**Explanation Sources**

* Tag similarity
* Genre overlap
* Collaborative neighborhood similarity
* Global consensus (critically acclaimed but under‑watched)

**Example Explanation**

> "Because you liked *Inception* and *The Matrix*, which share the tags *sci‑fi* and *mind‑bending*."

---

## 🌍 Diversity & Novelty Strategy

To mitigate popularity bias ("Harry Potter Effect"), ReelSense applies:

* Popularity‑normalized penalties
* Re‑ranking for long‑tail exposure
* Intra‑list diversity optimization

This ensures recommendations are **accurate, diverse, and non‑repetitive**.

---

## 📊 Evaluation Metrics (As per BrainDead Guidelines)

### A. Rating Prediction

* RMSE
* MAE

### B. Top‑K Recommendation (K = 10)

* Precision@10
* Recall@10
* NDCG@10
* MAP@10

### C. Diversity & Novelty

* Catalog Coverage
* Intra‑List Diversity
* Popularity‑Normalized Hits

Baseline models are used for comparative benchmarking.

---

## 🏗️ System Architecture

The ReelSense system is designed with a clear separation between **offline model training** and **online inference**, ensuring scalability, interpretability, and low-latency recommendations.

````mermaid
graph LR

    %% User Interaction Layer
    U[User] -->|Preferences & History| FE[Frontend UI]
    FE -->|REST Request| API[FastAPI Backend]

    %% Online Inference Pipeline
    subgraph Online Inference
        API --> PROF[User Profile Builder]
        PROF --> AGG[Hybrid Score Aggregator]
        AGG --> RERANK[Diversity Re-Ranker]
        RERANK --> EXPL[Explainability Engine]
    end

    %% Offline Training Pipeline
    subgraph Offline Training
        DATA[(MovieLens Dataset)] --> PRE[Preprocessing & EDA]
        PRE --> SVD[SVD Collaborative Model]
        PRE --> CONT[Content Model (Genres + Tags)]
        PRE --> BAY[Bayesian Popularity Smoothing]
    end

    %% Model Integration
    SVD --> AGG
    CONT --> AGG
    BAY --> AGG

    %% Final Response
    EXPL -->|Top-K Movies + Explanations| API
    API --> FE
```mermaid
graph LR

    %% User Interaction Layer
    U[User] -->|Preferences / History| FE[Frontend UI]
    FE -->|API Request| API[FastAPI Service]

    %% Online Inference Pipeline
    subgraph Online Inference
        API --> PROF[User Profile Builder]
        PROF --> AGG[Hybrid Score Aggregator]
        AGG --> RERANK[Diversity Re-Ranker]
        RERANK --> EXPL[Explainability Engine]
    end

    %% Offline Training Pipeline
    subgraph Offline Training
        DATA[(MovieLens Dataset)] --> PRE[Preprocessing & EDA]
        PRE --> CF[Collaborative Model (SVD)]
        PRE --> CB[Content Model (Genres + Tags)]
        PRE --> POP[Popularity & Bayesian Smoothing]
    end

    %% Model Usage
    CF --> AGG
    CB --> AGG
    POP --> AGG

    %% Response
    EXPL -->|Top-K Movies + Explanations| API
    API --> FE
````

---

## 📦 Project Structure

```
ReelSense/
├── data/                 # Data loading & preprocessing scripts
├── notebooks/            # EDA & experiment notebooks
├── recommender/          # Recommendation models
├── explainability/       # Explanation generation logic
├── evaluation/           # Metric computation
├── app/                  # FastAPI backend
├── requirements.txt
└── README.md
```

---

## 📦 Deliverables Checklist (BrainDead)

* ✔️ Concise technical report (PDF / PPT)
* ✔️ Public GitHub repository
* ✔️ iPython notebooks for EDA & modeling
* ✔️ Model training and evaluation code
* ✔️ Explainability demonstrations

---

## 👥 Team Details

**Department of Computer Science and Technology**
**IIEST Shibpur**

* **Rachit** – Lead ML Engineer & System Architect
* **Sarvesh** – Frontend Developer & UI/UX
* **Atharva** – Data Analyst & Evaluation Specialist

---

## ❤️ Acknowledgements

Built with passion for **Revelation 2K26 – BrainDead**.
May our models generalize, our bias reduce, and our recommendations make sense.


