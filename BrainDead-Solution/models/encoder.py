
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from transformers import AutoTokenizer, AutoModel
import pandas as pd
import os
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms


# Data loading utilities for MIMIC-CXR dataset
class MIMICCXRDataset(Dataset):

    
    def __init__(self, csv_path, img_root_dir, transform=None, max_samples=None):

        self.csv_path = csv_path
        self.img_root_dir = Path(img_root_dir)
        self.transform = transform
        
        # Load CSV
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV file not found: {csv_path}")
        
        self.df = pd.read_csv(csv_path)
        
        if max_samples:
            self.df = self.df.iloc[:max_samples]
        
        # Filter to only existing images
        self.valid_indices = []
        for idx, row in self.df.iterrows():
            img_path = self._get_image_path(row)
            if img_path.exists():
                self.valid_indices.append(idx)
        
        print(f"Found {len(self.valid_indices)} valid images out of {len(self.df)}")
    
    def _get_image_path(self, row):
        # The 'image' column contains a list of image paths as a string
        # e.g., "['files/p10/p10000032/s50414267/02aa804e-bde0afdd-112c0b34-7bc16630-4e384014.jpg', ...]"
        if 'image' in row and pd.notna(row['image']):
            img_list_str = row['image']
            # Parse the string list
            import ast
            try:
                img_list = ast.literal_eval(img_list_str)
                if isinstance(img_list, list) and len(img_list) > 0:
                    # Use the first image in the list
                    img_file = img_list[0]
                else:
                    return Path()  # Invalid
            except:
                return Path()  # Failed to parse
        elif 'path' in row and pd.notna(row['path']):
            img_file = row['path']
        elif 'filename' in row and pd.notna(row['filename']):
            img_file = row['filename']
        else:
            return Path()  # Invalid path
        
        # Image paths are already relative from 'files/', so just prepend the root
        full_path = self.img_root_dir / 'official_data_iccv_final' / img_file
        return full_path
    
    def __len__(self):
        return len(self.valid_indices)
    
    def __getitem__(self, idx):
        actual_idx = self.valid_indices[idx]
        row = self.df.iloc[actual_idx]
        
        img_path = self._get_image_path(row)
        
        # Load image
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Failed to load {img_path}: {e}")
            # Return dummy image on error
            image = Image.new('RGB', (224, 224))
        
        if self.transform:
            image = self.transform(image)
        
        # Get all available labels including CheXpert labels
        labels = {}
        
        # CheXpert labels
        chexpert_labels = [
            'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema',
            'Enlarged Cardiomediastinum', 'Fracture', 'Lung Lesion', 'Lung Opacity',
            'Pleural Effusion', 'Pleural Other', 'Pneumonia', 'Pneumothorax',
            'Support Devices', 'No Finding'
        ]
        
        for label_name in chexpert_labels:
            if label_name in row.index and pd.notna(row[label_name]):
                labels[label_name] = torch.tensor(row[label_name], dtype=torch.long)
            else:
                labels[label_name] = torch.tensor(0, dtype=torch.long)  # Default to 0 if missing
        
        # Add text/report if available
        if 'text' in row and pd.notna(row['text']):
            labels['text'] = row['text']
        elif 'report' in row and pd.notna(row['report']):
            labels['report'] = row['report']
        
        return image, labels


def get_mimic_cxr_loaders(archive_root, batch_size=2, max_samples=None):

    # Memory-efficient transforms for medical images
    # X-rays are grayscale but models expect 3 channels
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),  # Convert to single channel first
        transforms.Resize((128, 128)),  # Smaller resolution to save RAM (64x reduction vs 224x224)
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.shape[0] == 1 else x),  # Duplicate to 3 channels for pretrained models
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    train_csv = os.path.join(archive_root, 'mimic_cxr_aug_train_labeled.csv')
    val_csv = os.path.join(archive_root, 'mimic_cxr_aug_validate_labeled.csv')
    
    train_dataset = MIMICCXRDataset(train_csv, archive_root, transform, max_samples)
    val_dataset = MIMICCXRDataset(val_csv, archive_root, transform, max_samples)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    return train_loader, val_loader


RADLEX_TERMS = {
    'anatomical_regions': [
        'right lung', 'left lung', 'right upper lobe', 'right middle lobe', 
        'right lower lobe', 'left upper lobe', 'left lower lobe', 'lingula',
        'cardiac silhouette', 'mediastinum', 'hilum', 'right hilum', 'left hilum',
        'costophrenic angle', 'cardiophrenic angle', 'hemidiaphragm', 
        'right hemidiaphragm', 'left hemidiaphragm', 'trachea', 'carina',
        'aortic arch', 'aortic knob', 'pulmonary artery', 'superior vena cava',
        'chest wall', 'soft tissue', 'bony thorax', 'ribs', 'clavicle', 'scapula'
    ],
    
    'pathological_findings': [
        'pneumonia', 'consolidation', 'infiltrate', 'opacity', 'atelectasis',
        'pleural effusion', 'pulmonary edema', 'cardiomegaly', 'nodule', 
        'mass', 'cavity', 'pneumothorax', 'hemothorax', 'emphysema',
        'fibrosis', 'interstitial lung disease', 'granuloma', 'calcification',
        'tuberculosis', 'bronchiectasis', 'pulmonary embolism', 'infarction',
        'abscess', 'cyst', 'bullae', 'bleb', 'bronchiolitis'
    ],
    
    'abnormalities': [
        'ground-glass opacity', 'reticular pattern', 'nodular pattern',
        'reticulonodular pattern', 'honeycombing', 'tree-in-bud pattern',
        'air bronchogram', 'silhouette sign', 'kerley b lines', 'air crescent sign',
        'halo sign', 'reverse halo sign', 'crazy-paving pattern', 'mosaic attenuation',
        'bronchial wall thickening', 'peribronchial cuffing', 'pulmonary congestion',
        'vascular engorgement', 'hilar prominence', 'mediastinal widening'
    ],
    
    'cardiac_findings': [
        'cardiomegaly', 'cardiac enlargement', 'left atrial enlargement',
        'right atrial enlargement', 'left ventricular hypertrophy',
        'right ventricular hypertrophy', 'pericardial effusion', 
        'calcified aorta', 'tortuous aorta', 'dilated aorta',
        'prominent pulmonary artery', 'cardiac decompensation'
    ],
    
    'pleural_findings': [
        'pleural effusion', 'pleural thickening', 'pleural plaques',
        'pneumothorax', 'hemothorax', 'hydropneumothorax', 'empyema',
        'pleural mass', 'pleural nodule', 'loculated effusion',
        'blunted costophrenic angle', 'meniscus sign'
    ],
    
    'airway_findings': [
        'bronchiectasis', 'bronchial wall thickening', 'mucus plugging',
        'airway obstruction', 'tracheal deviation', 'tracheal narrowing',
        'bronchial stenosis', 'tracheobronchial tree', 'bronchial dilatation'
    ],
    
    'vascular_findings': [
        'pulmonary vascular congestion', 'pulmonary hypertension',
        'pulmonary embolism', 'vascular engorgement', 'arteriovenous malformation',
        'pulmonary artery enlargement', 'aortic dissection', 'aortic aneurysm'
    ],
    
    'descriptors': [
        'focal', 'diffuse', 'bilateral', 'unilateral', 'peripheral', 'central',
        'upper zone', 'middle zone', 'lower zone', 'basal', 'apical',
        'segmental', 'lobar', 'patchy', 'confluent', 'scattered',
        'linear', 'round', 'oval', 'irregular', 'well-defined', 'ill-defined',
        'homogeneous', 'heterogeneous', 'dense', 'faint', 'subtle'
    ],
    
    'procedures_devices': [
        'endotracheal tube', 'nasogastric tube', 'central venous catheter',
        'chest tube', 'pacemaker', 'implantable cardioverter defibrillator',
        'prosthetic valve', 'surgical clips', 'sternotomy wires',
        'tracheostomy tube', 'pulmonary artery catheter'
    ]
}


def get_radlex_terms(categories=None):

    if categories is None:
        # Return all terms
        all_terms = []
        for terms in RADLEX_TERMS.values():
            all_terms.extend(terms)
        return all_terms
    
    # Return specific categories
    terms = []
    for cat in categories:
        if cat in RADLEX_TERMS:
            terms.extend(RADLEX_TERMS[cat])
    return terms


class ProFAEncoder(nn.Module):

    def __init__(self, num_regions=6, device="cpu"):
        super().__init__()
        self.device = device
        self.num_regions = num_regions
        self.hidden_dim = 256
        self.radlex_dim = 768  # BioClinicalBERT output dimension
        
        # 1. Load ConvNeXt-Tiny from timm
        self.backbone = timm.create_model('convnext_tiny', pretrained=True)
        
        # Extract the feature extraction layer names for intermediate outputs
        # ConvNeXt-Tiny has 4 stages
        self.stage1 = nn.Sequential(*list(self.backbone.stem.children()))  # Stem layer
        self.stage2 = self.backbone.stages[0]  # Stage 1
        self.stage3 = self.backbone.stages[1]  # Stage 2
        self.stage4_1 = self.backbone.stages[2]  # Stage 3
        self.stage4_2 = self.backbone.stages[3]  # Stage 4
        
        # 2. Freeze backbone parameters
        for param in self.backbone.parameters():
            param.requires_grad = False
            
        # Get the output dimensions from ConvNeXt-Tiny stages
        # ConvNeXt-Tiny stages: stem -> 64, stage0 -> 96, stage1 -> 192, stage2 -> 384, stage3 -> 768
        self.stage1_dim = 96      # Output from stage 0 (pixel-level)
        self.stage3_dim = 384     # Output from stage 2 (region-level)
        self.stage4_dim = 768     # Output from stage 3 (organ-level)
        
        # 3. Define learnable region queries (region tokens)
        self.region_tokens = nn.Parameter(
            torch.randn(1, num_regions, self.hidden_dim)
        )
        nn.init.xavier_uniform_(self.region_tokens)
        
        # 4. Define MultiheadAttention layer for region-level feature extraction
        self.region_attention = nn.MultiheadAttention(
            embed_dim=self.hidden_dim,
            num_heads=8,
            batch_first=True,
            dropout=0.1
        )
        
        # 5. Define projection layers (to 256 dim)
        # Project stage1 features (pixel-level) to hidden_dim
        self.pixel_proj = nn.Sequential(
            nn.Conv2d(self.stage1_dim, self.hidden_dim, kernel_size=1),
            nn.BatchNorm2d(self.hidden_dim)
        )
        
        # Project stage3 features (region-level) to hidden_dim
        self.region_proj = nn.Sequential(
            nn.Conv2d(self.stage3_dim, self.hidden_dim, kernel_size=1),
            nn.BatchNorm2d(self.hidden_dim)
        )
        
        # Project stage4 features (organ-level) to hidden_dim
        self.organ_proj = nn.Linear(self.stage4_dim, self.hidden_dim)
        
        # 6. Load BioClinicalBERT (frozen) with fallback
        try:
            # Try the correct Bio_ClinicalBERT identifier
            self.radlex_tokenizer = AutoTokenizer.from_pretrained(
                "emilyalsentzer/Bio_ClinicalBERT"
            )
            self.radlex_model = AutoModel.from_pretrained(
                "emilyalsentzer/Bio_ClinicalBERT"
            )
        except OSError:
            # Fallback to standard BERT if Bio_ClinicalBERT unavailable
            print("Warning: Bio_ClinicalBERT not found. Using standard BERT as fallback.")
            self.radlex_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
            self.radlex_model = AutoModel.from_pretrained("bert-base-uncased")
        
        # Freeze BERT model
        for param in self.radlex_model.parameters():
            param.requires_grad = False
        
        # 7. Define RadLex projection layer (from 768 to hidden_dim)
        self.radlex_proj = nn.Linear(self.radlex_dim, self.hidden_dim)
        
        # Cross-attention between RadLex embeddings and spatial features
        self.radlex_attention = nn.MultiheadAttention(
            embed_dim=self.hidden_dim,
            num_heads=8,
            batch_first=True,
            dropout=0.1
        )
        
        # Layer normalization for stability
        self.norm1 = nn.LayerNorm(self.hidden_dim)
        self.norm2 = nn.LayerNorm(self.hidden_dim)

    def unfreeze_backbone(self):

        for param in self.backbone.parameters():
            param.requires_grad = True
        print("✓ ConvNeXt backbone unfrozen - all parameters trainable")
    
    def freeze_backbone(self):

        for param in self.backbone.parameters():
            param.requires_grad = False
        print("✓ ConvNeXt backbone frozen")
    
    def unfreeze_radlex(self):

        for param in self.radlex_model.parameters():
            param.requires_grad = True
        print("✓ BioClinicalBERT unfrozen - all parameters trainable")
    
    def freeze_radlex(self):
        for param in self.radlex_model.parameters():
            param.requires_grad = False
        print("✓ BioClinicalBERT frozen")
    
    def unfreeze_all(self):
        self.unfreeze_backbone()
        self.unfreeze_radlex()
        for param in self.parameters():
            param.requires_grad = True
        print("✓ All parameters unfrozen - full model fine-tuning enabled")
        self._print_trainable_params()
    
    def _print_trainable_params(self):
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen_params = total_params - trainable_params
        print(f"  - Total parameters: {total_params:,}")
        print(f"  - Trainable parameters: {trainable_params:,}")
        print(f"  - Frozen parameters: {frozen_params:,}")

    def encode_radlex(self, terms):

        if isinstance(terms, str):
            terms = [terms]
        
        # Tokenize and encode with BioClinicalBERT
        with torch.no_grad():
            encoded = self.radlex_tokenizer(
                terms,
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=128
            )
            encoded = {k: v.to(self.device) for k, v in encoded.items()}
            
            # Get [CLS] token representation (pooled output)
            radlex_output = self.radlex_model(**encoded)
            radlex_embeddings = radlex_output.pooler_output  # Shape: (N, 768)
        
        return radlex_embeddings

    def forward(self, images, radlex_terms=None):

        batch_size = images.size(0)
        
        # 1. Extract multi-scale features from ConvNeXt backbone
        # Stage 1: Stem -> high-resolution (B, 64, H/4, W/4)
        x = self.stage1(images)
        pixel_level = self.stage2(x)  # (B, 64, H/4, W/4)
        
        # Stage 3: intermediate layers -> (B, 256, H/16, W/16)
        x = self.stage3(pixel_level)
        region_level = self.stage4_1(x)  # (B, 256, H/16, W/16)
        
        # Stage 4: final layer -> (B, 512, H/32, W/32)
        organ_level = self.stage4_2(region_level)  # (B, 512, H/32, W/32)
        
        # 2. Extract pixel-level features from early stage
        pixel_features = self.pixel_proj(pixel_level)  # (B, 256, H/4, W/4)
        # Flatten spatial dimensions for sequence
        B, C, H, W = pixel_features.shape
        pixel_features = pixel_features.view(B, C, -1).permute(0, 2, 1)  # (B, N, 256)
        
        # 3. Extract region-level features via cross-attention
        # Project region features to hidden_dim
        region_proj = self.region_proj(region_level)  # (B, 256, H/16, W/16)
        B, C, H_r, W_r = region_proj.shape
        region_proj_flat = region_proj.view(B, C, -1).permute(0, 2, 1)  # (B, N_r, 256)
        
        # Apply cross-attention: region tokens attend over spatial features
        region_tokens = self.region_tokens.expand(B, -1, -1)  # (B, num_regions, 256)
        region_features, _ = self.region_attention(
            region_tokens,
            region_proj_flat,
            region_proj_flat
        )  # (B, num_regions, 256)
        region_features = self.norm1(region_features)
        
        # 4. Extract organ-level feature via global pooling
        organ_pooled = F.adaptive_avg_pool2d(organ_level, 1)  # (B, 512, 1, 1)
        organ_pooled = organ_pooled.view(B, -1)  # (B, 512)
        organ_feature = self.organ_proj(organ_pooled)  # (B, 256)
        
        # 5. Perform optional RadLex alignment
        radlex_alignment = None
        if radlex_terms is not None:
            # Encode RadLex terms
            radlex_embeddings = self.encode_radlex(radlex_terms)  # (N_terms, 768)
            radlex_embeddings = self.radlex_proj(radlex_embeddings)  # (N_terms, 256)
            
            # Expand for batch: replicate across batch
            radlex_embeddings = radlex_embeddings.unsqueeze(0).expand(B, -1, -1)  # (B, N_terms, 256)
            
            # Cross-attention between RadLex and region features
            radlex_alignment, _ = self.radlex_attention(
                region_features,
                radlex_embeddings,
                radlex_embeddings
            )  # (B, num_regions, 256)
            radlex_alignment = self.norm2(radlex_alignment)
        
        # 6. Return all feature levels
        return {
            'pixel_features': pixel_features,      # (B, N, 256)
            'region_features': region_features,    # (B, num_regions, 256)
            'organ_feature': organ_feature,        # (B, 256)
            'radlex_alignment': radlex_alignment   # (B, num_regions, 256) or None
        }


def cross_scale_consistency(region_features, organ_feature):

    # Aggregate region features by averaging
    region_aggregated = region_features.mean(dim=1)  # (B, 256)
    
    # Compute cosine similarity between aggregated regions and organ feature
    # Normalize both vectors
    region_aggregated_norm = F.normalize(region_aggregated, p=2, dim=1)  # (B, 256)
    organ_feature_norm = F.normalize(organ_feature, p=2, dim=1)  # (B, 256)
    
    # Compute cosine similarity (dot product of normalized vectors)
    consistency = torch.sum(region_aggregated_norm * organ_feature_norm, dim=1)  # (B,)
    
    # Return mean consistency across batch
    return consistency.mean()
