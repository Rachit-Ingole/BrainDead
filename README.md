# 🧠 Pookies - BrainDead @ Revelation 2K26

> **Complete Solutions Repository for BrainDead Competition**
> Department of Computer Science and Technology, IIEST Shibpur

This repository contains comprehensive solutions for **both Problem Statements** of the BrainDead competition at Revelation 2K26.

---

## 📋 Table of Contents

- [Project Overview](#-project-overview)
- [Problem Statement 1: ReelSense](#-problem-statement-1-reelsense)
- [Problem Statement 2: BrainDead-Solution](#-problem-statement-2-braindead-solution)
- [Repository Structure](#-repository-structure)
- [Quick Start Guide](#-quick-start-guide)
- [Contributors](#-contributors)
- [Acknowledgements](#-acknowledgements)

---

## 🎯 Project Overview

**Pookies** is a dual-project repository showcasing advanced machine learning systems:

1. **ReelSense**: An explainable movie recommender system with diversity optimization
2. **BrainDead-Solution**: A cognitive radiology system for automated chest X-ray analysis

Both projects demonstrate state-of-the-art approaches in their respective domains, combining deep learning, natural language processing, and explainable AI techniques.

---

## 🎬 Problem Statement 1: ReelSense

### Explainable Movie Recommender System with Diversity Optimization

**Objective**: Design a Top-K Movie Recommendation System that generates personalized recommendations while ensuring diversity and catalog coverage.

### Key Features

- **Hybrid Recommendation Engine**
  - Collaborative Filtering (SVD)
  - Content-based filtering (genres + tags)
  - Popularity-based smoothing
  
- **Diversity Optimization**
  - Mitigates popularity bias
  - Ensures catalog coverage
  - Balances relevance with novelty

- **Explainability**
  - Natural language explanations for each recommendation
  - Transparent decision-making process

### Dataset

**MovieLens Latest Small**
- 100,836 ratings
- 610 users
- 9,742 movies

### Tech Stack

- Python 3.8+
- scikit-surprise (Collaborative Filtering)
- FastAPI (Backend)
- React + Vite (Frontend)
- pandas, numpy (Data processing)

### Quick Start

```bash
cd ReelSense

# Backend setup
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
pip install -r requirements.txt
python api.py

# Frontend setup (separate terminal)
cd frontend
npm install
npm run dev
```

Access the application at `http://localhost:5173`

### Performance Metrics

- **Ranking Metrics**: NDCG@K, Precision@K, Recall@K
- **Diversity Metrics**: Intra-list Diversity, Coverage
- **Novelty Metrics**: Surprise factor based on popularity

---

## 🏥 Problem Statement 2: BrainDead-Solution

### Cognitive Radiology System for Automated Chest X-Ray Analysis

**Objective**: Build an end-to-end system that analyzes chest X-rays to generate disease classifications and clinical reports.

This system implements a **cognitive simulation architecture** for generating structured chest X-ray radiology reports. Unlike simple image captioning models, this system simulates the actual cognitive workflow of a radiologist.

### The Three Mandatory Modules

#### **Module 1: PRO-FA (Hierarchical Visual Alignment)**
- **File**: `models/profa.py`
- **Purpose**: Extract hierarchical visual features from X-rays
- **Architecture**:
  - ConvNeXt-Tiny backbone (frozen)
  - Multi-scale feature extraction:
    - Pixel-level (fine textures)
    - Region-level (anatomical regions)
    - Organ-level (global view)
  - RadLex medical terminology alignment via BioClinicalBERT
- **Outputs**:
  - `pixel_features`: (B, N, 256)
  - `region_features`: (B, R, 256)
  - `organ_feature`: (B, 256)

#### **Module 2: MIX-MLP (Knowledge-Enhanced Classification)**
- **File**: `models/classifier.py`
- **Purpose**: Predict disease probabilities before report generation
- **Architecture**:
  - Dual-path MLP design:
    - Residual Path: Stable shallow mapping
    - Expansion Path: Models disease co-occurrence
  - Predicts CheXpert 14 labels
- **Outputs**:
  - `disease_logits`: (B, 14)
  - `disease_probs`: (B, 14)

#### **Module 3: RCTA (Triangular Cognitive Attention)**
- **File**: `models/decoder.py`
- **Purpose**: Generate structured reports via cognitive attention
- **Architecture**:
  - Three-stage triangular attention:
    1. Image → Clinical Context
    2. Context → Disease Hypothesis
    3. Hypothesis → Image Verification
  - Lightweight Transformer decoder (2 layers)
  - Separate heads for Findings and Impression
- **Outputs**:
  - `findings_logits`: (B, T1, vocab_size)
  - `impression_logits`: (B, T2, vocab_size)

### Architecture Overview

#### 1. PRO-FA (Progressive Region-based Feature Aggregation)
- **Purpose**: Hierarchical feature extraction from chest X-ray images
- **Input**: Raw chest X-ray images (128×128)
- **Output**: Region-based features (7×7 grid) + global organ-level features
- **Architecture**: ResNet-50 backbone with progressive attention mechanisms

#### 2. MIX-MLP (Multi-scale Interactive eXpert MLP)
- **Purpose**: Multi-label disease classification
- **Input**: Region features from PRO-FA
- **Output**: 14 CheXpert disease probabilities
- **Architecture**: Multi-scale MLP with cross-attention to region features

#### 3. RCTA (Region-aware Cognitive Text Attention)
- **Purpose**: Clinical report generation
- **Input**: Image features + disease predictions + clinical context
- **Output**: Structured clinical reports (findings + impression)
- **Architecture**: Transformer decoder with triangular attention mechanisms

*Note: Models trained for limited epochs (6) on CPU. Performance expected to improve with GPU training and more epochs.*

### Tech Stack

- Python 3.8+
- PyTorch (Deep Learning)
- Streamlit (Web Interface)
- Docker (Deployment)
- OpenCV (Image Processing)

### Quick Start

#### Docker Deployment (Recommended)

```bash
cd BrainDead-Solution
./docker-run.sh build && ./docker-run.sh run
# Open http://localhost:8501
```

#### Manual Setup

```bash
cd BrainDead-Solution
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
pip install -r requirements.txt
streamlit run app.py
# Open http://localhost:8501
```

### Web Interface Features

- 📤 Drag & drop X-ray image upload
- 🏥 14-label CheXpert disease classification
- 📝 Automated clinical report generation
- ⚡ Real-time processing with progress indicators
- 📊 Interactive visualization


---

## 📂 Repository Structure

```
Pookies/
├── ReelSense/                      # Movie Recommendation System (PS1)
│   ├── api.py                      # FastAPI backend
│   ├── train.py                    # Model training pipeline
│   ├── model_training.ipynb        # Training notebook
│   ├── requirements.txt
│   ├── frontend/                   # React UI
│   │   ├── src/
│   │   ├── index.html
│   │   ├── package.json
│   │   └── vite.config.js
│   └── README.md                   # Detailed documentation
│
├── BrainDead-Solution/             # Medical Imaging System (PS2)
│   ├── app.py                      # Streamlit web interface
│   ├── Dockerfile                  # Container configuration
│   ├── docker-compose.yml
│   ├── docker-run.sh              # Deployment script
│   ├── requirements.txt
│   ├── models/                     # Model architecture
│   │   ├── encoder.py             # PRO-FA implementation
│   │   ├── classifier.py          # MIX-MLP implementation
│   │   └── decoder.py             # RCTA implementation
│   ├── training/
│   │   └── train_full_system.py   # End-to-end training
│   ├── evaluation/
│   │   └── evaluate_iu_xray.py    # Metrics computation
│   ├── notebooks/
│   │   └── inference_demo.ipynb   # Demo notebook
│   ├── checkpoints/                # Pre-trained models
│   └── README.md                   # Detailed documentation
│
├── archive (1)/                    # MIMIC-CXR dataset
├── archive (2)/                    # Indiana University dataset
├── chexpert-labeler/              # CheXpert labeling tool
├── data/                          # Shared data utilities
├── models/                        # Shared model components
├── checkpoints/                   # Shared checkpoints
└── README.md                      # This file
```

---

## 🚀 Quick Start Guide

### Prerequisites

- Python 3.8 or higher
- Node.js 16+ (for ReelSense frontend)
- Docker (optional, for BrainDead-Solution)
- Git

### Clone Repository

```bash
git clone <repository-url>
cd Pookies
```

### Setup ReelSense

```bash
cd ReelSense
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Start backend
python api.py &

# Start frontend (separate terminal)
cd frontend
npm install
npm run dev
```

Access ReelSense at `http://localhost:5173`

### Setup BrainDead-Solution

```bash
cd BrainDead-Solution

# Option 1: Docker (Recommended)
./docker-run.sh build && ./docker-run.sh run

# Option 2: Manual
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Access BrainDead-Solution at `http://localhost:8501`

---

## 👥 Contributors

### Team Pookies

- **Rachit** – Lead ML Engineer & System Architect
  - Project lead for both ReelSense and BrainDead-Solution
  - Deep learning architecture design
  - System integration and deployment
  
- **Sarvesh** – Frontend Developer & UI/UX
  - ReelSense web interface development
  - User experience design
  - Frontend architecture
  
- **Atharva** – Data Analyst & Evaluation Specialist
  - Data preprocessing and EDA
  - Metrics computation and analysis
  - Model evaluation and benchmarking


## 🙏 Acknowledgements

Built with passion for **Revelation 2K26 – BrainDead** competition.

**Last Updated**: February 2026
