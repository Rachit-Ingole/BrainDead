"""
End-to-End Training for Complete Cognitive Radiology System
============================================================

Trains all three modules together:
1. PRO-FA (Encoder) - Visual feature extraction
2. MIX-MLP (Classifier) - Disease classification
3. RCTA (Decoder) - Report generation

This is the proper way to train - all modules learn together with gradients
flowing through the entire pipeline.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from models.encoder import ProFAEncoder, get_mimic_cxr_loaders
from models.classifier import MixMLPClassifier
from models.decoder import RCTA
from transformers import GPT2Tokenizer
from tqdm import tqdm
import os

# CheXpert label names
CHEXPERT_LABELS = [
    'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema',
    'Enlarged Cardiomediastinum', 'Fracture', 'Lung Lesion', 'Lung Opacity',
    'Pleural Effusion', 'Pleural Other', 'Pneumonia', 'Pneumothorax',
    'Support Devices', 'No Finding'
]

class CognitiveRadiologySystem(nn.Module):
    """
    Complete end-to-end system combining all three modules.
    """
    def __init__(self, device='cpu'):
        super().__init__()
        
        # Module 1: PRO-FA Encoder
        self.encoder = ProFAEncoder(num_regions=6, device=device)
        
        # Module 2: MIX-MLP Classifier
        self.classifier = MixMLPClassifier(
            input_dim=256,  # From encoder organ_feature
            num_labels=14  # CheXpert labels
        )
        
        # Module 3: RCTA Decoder
        self.decoder = RCTA(
            embed_dim=256,
            num_heads=8,
            vocab_size=50257,  # GPT-2 vocab size
            max_seq_len=128,
            num_decoder_layers=2,
            dropout=0.1
        )
        
        self.device = device
    
    def forward(self, images, report_tokens=None):
        """
        Forward pass through all three modules.
        
        Args:
            images: (B, 3, 224, 224) - Chest X-ray images
            report_tokens: (B, L) - Target report tokens (for training)
        
        Returns:
            dict with:
                - encoder_output: Dict from encoder
                - disease_logits: (B, 14) disease predictions
                - disease_probs: (B, 14) disease probabilities
                - findings_logits: (B, L, vocab_size) if training
                - impression_logits: (B, L, vocab_size) if training
        """
        # Module 1: Extract visual features
        encoder_output = self.encoder(images)
        
        # Module 2: Predict diseases
        classifier_output = self.classifier(
            encoder_output['organ_feature'],
            encoder_output['region_features']
        )
        
        # Module 3: Generate report using triangular attention
        if report_tokens is not None:
            # Training mode - use same tokens for findings and impression
            # (simplified - in practice you'd split the report)
            decoder_output = self.decoder(
                region_features=encoder_output['region_features'],
                organ_feature=encoder_output['organ_feature'],
                disease_logits=classifier_output['disease_logits'],
                findings_tokens=report_tokens,
                impression_tokens=report_tokens,
                clinical_text_ids=None
            )
        else:
            # Inference mode
            decoder_output = self.decoder.generate(
                region_features=encoder_output['region_features'],
                organ_feature=encoder_output['organ_feature'],
                disease_logits=classifier_output['disease_logits'],
                max_length=128
            )
        
        return {
            'encoder_output': encoder_output,
            'disease_logits': classifier_output['disease_logits'],
            'disease_probs': classifier_output['disease_probs'],
            'findings_logits': decoder_output.get('findings_logits'),
            'impression_logits': decoder_output.get('impression_logits'),
            'verified_representation': decoder_output.get('verified_representation')
        }


def prepare_batch(images, labels_dict, tokenizer, device):
    """
    Prepare a batch for training.
    
    Returns:
        images: Tensor on device
        disease_labels: (B, 14) binary labels
        report_tokens: (B, L) tokenized reports
    """
    images = images.to(device)
    
    # Extract CheXpert labels
    batch_size = images.size(0)
    disease_labels = torch.zeros(batch_size, len(CHEXPERT_LABELS))
    
    for i, label_name in enumerate(CHEXPERT_LABELS):
        if label_name in labels_dict:
            label_values = labels_dict[label_name]
            # Convert -1 (uncertain) to 0
            label_values = torch.where(label_values == -1, 
                                      torch.zeros_like(label_values), 
                                      label_values)
            disease_labels[:, i] = label_values.float()
    
    disease_labels = disease_labels.to(device)
    
    # Tokenize reports if available
    report_tokens = None
    if 'text' in labels_dict:
        reports = labels_dict['text']
        # Handle both string and list of strings
        if isinstance(reports, str):
            reports = [reports]
        
        # Tokenize
        encoded = tokenizer(
            reports,
            padding='max_length',
            truncation=True,
            max_length=128,
            return_tensors='pt'
        )
        report_tokens = encoded['input_ids'].to(device)
    
    return images, disease_labels, report_tokens


def compute_loss(output, disease_labels, report_tokens, tokenizer):
    """
    Compute combined loss for all three modules.
    
    Loss = α * classification_loss + β * generation_loss
    """
    # Classification loss (BCE for multi-label)
    classification_loss = nn.BCEWithLogitsLoss()(
        output['disease_logits'], 
        disease_labels
    )
    
    # Generation loss (Cross-entropy for next token prediction)
    generation_loss = 0.0
    if output['findings_logits'] is not None and report_tokens is not None:
        # Compute loss for findings (using report tokens)
        logits = output['findings_logits'][:, :-1, :]  # (B, L-1, vocab)
        targets = report_tokens[:, 1:]  # (B, L-1)
        
        # Flatten for cross-entropy
        logits_flat = logits.reshape(-1, logits.size(-1))
        targets_flat = targets.reshape(-1)
        
        findings_loss = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)(
            logits_flat, 
            targets_flat
        )
        
        # Compute loss for impression (using same tokens for now)
        logits_imp = output['impression_logits'][:, :-1, :]
        logits_imp_flat = logits_imp.reshape(-1, logits_imp.size(-1))
        
        impression_loss = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)(
            logits_imp_flat,
            targets_flat
        )
        
        generation_loss = (findings_loss + impression_loss) / 2.0
    
    # Combine losses
    alpha = 1.0  # Classification weight
    beta = 0.5   # Generation weight
    
    total_loss = alpha * classification_loss + beta * generation_loss
    
    return {
        'total': total_loss,
        'classification': classification_loss,
        'generation': generation_loss
    }


# ============================================================================
# Main Training Script
# ============================================================================

if __name__ == "__main__":
    # Setup
    ARCHIVE_ROOT = os.path.join(os.path.dirname(__file__), 'archive (1)')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("=" * 70)
    print("End-to-End Cognitive Radiology System Training")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"Training all 3 modules together:")
    print("  1. PRO-FA (Encoder)")
    print("  2. MIX-MLP (Classifier)")
    print("  3. RCTA (Decoder)")
    
    # Initialize tokenizer
    print("\n[1] Initializing tokenizer...")
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    print(f"    ✓ GPT-2 tokenizer loaded (vocab size: {len(tokenizer)})")
    
    # Load data
    print("\n[2] Loading MIMIC-CXR dataset...")
    train_loader, val_loader = get_mimic_cxr_loaders(
        ARCHIVE_ROOT, 
        batch_size=8,  # Can use larger batch with compressed images
        max_samples=None  # Full dataset - images are now 128x128 (64x less RAM)
    )
    print(f"    ✓ Train batches: {len(train_loader)}")
    print(f"    ✓ Val batches: {len(val_loader)}")
    
    # Initialize model
    print("\n[3] Initializing complete system...")
    model = CognitiveRadiologySystem(device=device)
    model = model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"    - Total parameters: {total_params:,}")
    print(f"    - Encoder parameters: {sum(p.numel() for p in model.encoder.parameters()):,}")
    print(f"    - Classifier parameters: {sum(p.numel() for p in model.classifier.parameters()):,}")
    print(f"    - Decoder parameters: {sum(p.numel() for p in model.decoder.parameters()):,}")
    
    # Unfreeze encoder for end-to-end training
    print("\n[4] Unfreezing encoder for end-to-end training...")
    model.encoder.unfreeze_all()
    trainable_after = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"    ✓ All parameters trainable: {trainable_after:,}")
    
    # Setup training
    print("\n[5] Setting up training...")
    optimizer = optim.AdamW(model.parameters(), lr=1e-5, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    
    epochs = 10
    best_val_loss = float('inf')
    
    os.makedirs('checkpoints', exist_ok=True)
    
    print(f"    - Optimizer: AdamW")
    print(f"    - Learning rate: 1e-5")
    print(f"    - Epochs: {epochs}")
    print(f"    - Batch size: 8")
    print(f"    - Image size: 128x128 (compressed)")
    print(f"    - Full dataset (~45K training images)")
    print(f"    - Checkpoints: checkpoints/")
    
    # Training loop
    print("\n[6] Starting end-to-end training...")
    print("=" * 70)
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        total_loss = 0
        total_cls_loss = 0
        total_gen_loss = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        for batch_idx, (images, labels_dict) in enumerate(pbar):
            # Prepare batch
            images, disease_labels, report_tokens = prepare_batch(
                images, labels_dict, tokenizer, device
            )
            
            # Forward pass
            output = model(images, report_tokens)
            
            # Compute loss
            losses = compute_loss(output, disease_labels, report_tokens, tokenizer)
            loss = losses['total']
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
            total_cls_loss += losses['classification'].item()
            if isinstance(losses['generation'], torch.Tensor):
                total_gen_loss += losses['generation'].item()
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'cls': f'{losses["classification"].item():.4f}',
                'gen': f'{losses["generation"].item():.4f}' if isinstance(losses['generation'], torch.Tensor) else '0.0'
            })
        
        avg_train_loss = total_loss / len(train_loader)
        avg_cls_loss = total_cls_loss / len(train_loader)
        avg_gen_loss = total_gen_loss / len(train_loader)
        
        # Validation phase
        model.eval()
        val_loss = 0
        val_cls_loss = 0
        val_gen_loss = 0
        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]  ")
            for images, labels_dict in pbar:
                images, disease_labels, report_tokens = prepare_batch(
                    images, labels_dict, tokenizer, device
                )
                
                output = model(images, report_tokens)
                losses = compute_loss(output, disease_labels, report_tokens, tokenizer)
                
                val_loss += losses['total'].item()
                val_cls_loss += losses['classification'].item()
                if isinstance(losses['generation'], torch.Tensor):
                    val_gen_loss += losses['generation'].item()
                
                pbar.set_postfix({
                    'loss': f'{losses["total"].item():.4f}',
                    'cls': f'{losses["classification"].item():.4f}'
                })
        
        avg_val_loss = val_loss / len(val_loader)
        avg_val_cls = val_cls_loss / len(val_loader)
        avg_val_gen = val_gen_loss / len(val_loader)
        
        # Update learning rate
        scheduler.step(avg_val_loss)
        
        print(f"\nEpoch {epoch+1}/{epochs} Summary:")
        print(f"  Train Loss: {avg_train_loss:.4f} (cls: {avg_cls_loss:.4f}, gen: {avg_gen_loss:.4f})")
        print(f"  Val Loss:   {avg_val_loss:.4f} (cls: {avg_val_cls:.4f}, gen: {avg_val_gen:.4f})")
        
        # Save checkpoint if best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            checkpoint_path = f'checkpoints/full_system_best_epoch{epoch+1}.pt'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'encoder_state_dict': model.encoder.state_dict(),
                'classifier_state_dict': model.classifier.state_dict(),
                'decoder_state_dict': model.decoder.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
            }, checkpoint_path)
            print(f"  ✓ Saved best model to: {checkpoint_path}")
        
        print("-" * 70)
    
    print("\n" + "=" * 70)
    print("✓ Training complete!")
    print("=" * 70)
    print("\nTraining Summary:")
    print(f"  - Epochs completed: {epochs}")
    print(f"  - Best validation loss: {best_val_loss:.4f}")
    print(f"  - Best model saved to: checkpoints/")
    print(f"  - Total parameters trained: {trainable_after:,}")
    print("\nAll three modules trained end-to-end with:")
    print("  - Shared gradients flowing through entire pipeline")
    print("  - Disease classification guiding visual features")
    print("  - Report generation using verified representations")
    print("=" * 70)
