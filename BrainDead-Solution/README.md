# BrainDead-Solution: Cognitive Radiology System

A modular deep learning system for automated chest X-ray analysis, generating both disease classifications and clinical reports.

## 🏗️ Architecture Overview

The system consists of three main modules that simulate the cognitive workflow of a radiologist:

### 1. PRO-FA (Progressive Region-based Feature Aggregation)
- **Purpose**: Hierarchical feature extraction from chest X-ray images
- **Input**: Raw chest X-ray images (128×128)
- **Output**: Region-based features (7×7 grid) + global organ-level features
- **Architecture**: ResNet-50 backbone with progressive attention mechanisms

### 2. MIX-MLP (Multi-scale Interactive eXpert MLP)
- **Purpose**: Multi-label disease classification
- **Input**: Region features from PRO-FA
- **Output**: 14 CheXpert disease probabilities
- **Architecture**: Multi-scale MLP with cross-attention to region features

### 3. RCTA (Region-aware Cognitive Text Attention)
- **Purpose**: Clinical report generation
- **Input**: Image features + disease predictions + clinical context
- **Output**: Structured clinical reports (findings + impression)
- **Architecture**: Transformer decoder with triangular attention mechanisms

## 📊 Performance Benchmarks

| Metric | Target | Current Best | Status |
|--------|--------|--------------|--------|
| CheXpert F1 | >0.5 | 0.326 | ⚠️ Below target |
| RadGraph F1 | >0.5 | 0.401 | ⚠️ Below target |
| CIDEr | >0.4 | 0.004 | ⚠️ Below target |
| BLEU-4 | >0.3 | 0.0403 | ⚠️ Below target |

*Note: Models trained for limited epochs (6) on CPU. Performance expected to improve with GPU training and more epochs.*

## 🚀 Quick Start

### One-Command Docker Deployment (Recommended)
```bash
git clone <your-repo-url>
cd BrainDead-Solution
./docker-run.sh build && ./docker-run.sh run
# Open http://localhost:8501 in your browser
```

### Manual Setup
```bash
# Clone repository
git clone <your-repo-url>
cd BrainDead-Solution

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# or
.venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt

# Run web interface
streamlit run app.py
# Open http://localhost:8501
```

## 🖥️ Web Interface Features

- **📤 Image Upload**: Drag & drop chest X-ray images (PNG/JPG)
- **🏥 Disease Classification**: 14-label CheXpert classification with confidence scores
- **📝 Report Generation**: Automated clinical reports with findings and impression sections
- **⚡ Real-time Processing**: Instant results with progress indicators
- **📊 Interactive Visualization**: Clean display of classifications and generated reports

## 🐳 Docker Deployment (Recommended)

For reproducible deployment and easy setup, use Docker:

### Quick Docker Setup
```bash
# Build and run with the management script
./docker-run.sh build
./docker-run.sh run

# Or use docker-compose
docker-compose up --build
```

### Manual Docker Commands
```bash
# Build the image
docker build -t braindead-solution .

# Run the container
docker run -d \
  --name braindead-solution-app \
  -p 8501:8501 \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/checkpoints:/app/checkpoints \
  braindead-solution

# Access the web app at http://localhost:8501
```

### Docker Management
```bash
# View logs
./docker-run.sh logs

# Open shell in container
./docker-run.sh shell

# Stop container
./docker-run.sh stop

# Clean up
./docker-run.sh clean
```

**Benefits:**
- ✅ Reproducible environment across all systems
- ✅ All dependencies pre-installed and isolated
- ✅ Easy deployment and scaling
- ✅ Volume mounting for data persistence
- ✅ Production-ready containerization

## 📚 Usage Options

### Web Interface (Easiest)
```bash
# Docker (recommended)
./docker-run.sh run

# Manual
streamlit run app.py
# Open http://localhost:8501
```

### Python API
```python
from models.encoder import PROFA
from models.classifier import MIXMLP
from models.decoder import RCTA

# Load models
encoder = PROFA()
classifier = MIXMLP()
decoder = RCTA()

# Process image
image = load_chest_xray("path/to/image.jpg")
region_features, organ_feature = encoder(image)
disease_logits = classifier(region_features)
report = decoder.generate_report(region_features, organ_feature, disease_logits)
```

### Command Line Evaluation
```bash
# Evaluate on IU-Xray benchmark
python evaluation/evaluate_iu_xray.py \
    --checkpoint checkpoints/full_system_best_epoch6.pt \
    --max_samples 200 \
    --batch_size 8
```

### Demo Notebook
Interactive examples in `notebooks/inference_demo.ipynb`

## 📁 Project Structure

```
BrainDead-Solution/
├── .dockerignore        # Docker build optimization
├── .gitignore          # Git ignore rules
├── Dockerfile          # Container build instructions
├── README.md           # This documentation
├── app.py              # Streamlit web interface
├── checkpoints/        # Trained model weights
├── data/               # Data download and preprocessing scripts
├── docker-compose.yml  # Multi-container orchestration
├── docker-run.sh       # Docker management script
├── evaluation/         # CheXpert and RadGraph evaluators
├── models/             # Core model implementations
│   ├── encoder.py      # PRO-FA implementation
│   ├── classifier.py   # MIX-MLP implementation
│   └── decoder.py      # RCTA attention & generator
├── notebooks/          # Demo notebooks (inference examples)
├── requirements.txt    # Python dependencies
└── training/           # Training loops and loss functions
```

## 🔧 Development

### Data Preparation

#### MIMIC-CXR Dataset
```bash
# Download (requires PhysioNet access)
python data/preprocess.py --dataset mimic-cxr --download

# Preprocess
python data/preprocess.py --dataset mimic-cxr --preprocess
```

#### IU-Xray Dataset
```bash
# Download and preprocess
python data/preprocess.py --dataset iu-xray --download
python data/preprocess.py --dataset iu-xray --preprocess
```

### Training
```bash
# Train the full system
python training/train_full_system.py
```

### Evaluation
```bash
# Evaluate on IU-Xray benchmark
python evaluation/evaluate_iu_xray.py \
    --checkpoint checkpoints/full_system_best_epoch6.pt \
    --max_samples 200 \
    --batch_size 8
```

## 🚀 Deployment

### Production Deployment
```bash
# Using docker-compose for production
docker-compose up -d --build

# Scale the service
docker-compose up -d --scale braindead-solution=3

# View logs
docker-compose logs -f
```

### Cloud Deployment
The containerized setup makes it easy to deploy to:
- **AWS ECS/Fargate**
- **Google Cloud Run**
- **Azure Container Instances**
- **Kubernetes**
- **Heroku** (with heroku.yml)

### API Extension
For programmatic access, the system can be extended with FastAPI endpoints for REST API deployment.

## 🏥 CheXpert Disease Labels

The system classifies 14 chest X-ray findings:

1. **Atelectasis** - Partial lung collapse
2. **Cardiomegaly** - Enlarged heart
3. **Consolidation** - Lung consolidation
4. **Edema** - Pulmonary edema
5. **Enlarged Cardiomediastinum** - Enlarged cardiomediastinal silhouette
6. **Fracture** - Bone fractures
7. **Lung Lesion** - Lung lesions/nodules
8. **Lung Opacity** - Lung opacities
9. **No Finding** - Normal chest X-ray
10. **Pleural Effusion** - Fluid in pleural space
11. **Pleural Other** - Other pleural abnormalities
12. **Pneumonia** - Pneumonia
13. **Pneumothorax** - Air in pleural space
14. **Support Devices** - Medical devices (tubes, lines)

## 📄 Citation

If you use this code in your research, please cite:

```bibtex
@article{braindead2024,
  title={BrainDead-Solution: A Cognitive Radiology System for Chest X-ray Analysis},
  author={Your Name},
  journal={arXiv preprint},
  year={2024}
}
```

## 📋 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **CheXpert Dataset**: For providing the multi-label classification benchmark
- **IU-Xray Dataset**: For report generation evaluation data
- **MIMIC-CXR Dataset**: For training data
- **PyTorch**: For the deep learning framework
- **Streamlit**: For the web interface framework