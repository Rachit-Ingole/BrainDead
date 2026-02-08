import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms
import sys
import os
from transformers import GPT2Tokenizer
import numpy as np

# Add models directory
sys.path.append(os.path.join(os.path.dirname(__file__), 'models'))

from encoder import ProFAEncoder
from classifier import MixMLPClassifier
from decoder import RCTA

# CheXpert labels
CHEXPERT_LABELS = [
    'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema',
    'Enlarged Cardiomediastinum', 'Fracture', 'Lung Lesion', 'Lung Opacity',
    'Pleural Effusion', 'Pleural Other', 'Pneumonia', 'Pneumothorax',
    'Support Devices', 'No Finding'
]

class CognitiveRadiologySystem(nn.Module):
    """Complete system for inference"""
    def __init__(self, device='cpu'):
        super().__init__()
        self.encoder = ProFAEncoder(num_regions=6, device=device)
        self.classifier = MixMLPClassifier(input_dim=256, num_labels=14)
        self.decoder = RCTA(
            embed_dim=256, num_heads=8, vocab_size=50257,
            max_seq_len=128, num_decoder_layers=2, dropout=0.1
        )
        self.device = device

    def forward(self, images, tokenizer=None):
        encoder_output = self.encoder(images)
        classifier_output = self.classifier(
            encoder_output['organ_feature'],
            encoder_output['region_features']
        )

        # Generate report
        start_id = tokenizer.bos_token_id if tokenizer and hasattr(tokenizer, 'bos_token_id') else 50256
        end_id = tokenizer.eos_token_id if tokenizer else 50256
        
        decoder_output = self.decoder.generate(
            region_features=encoder_output['region_features'],
            organ_feature=encoder_output['organ_feature'],
            disease_logits=classifier_output['disease_logits'],
            max_length=80,
            start_token_id=start_id,
            end_token_id=end_id,
            temperature=0.7,
            top_p=0.9
        )

        return {
            'disease_probs': classifier_output['disease_probs'],
            'generated_findings': decoder_output.get('findings_tokens'),
            'generated_impression': decoder_output.get('impression_tokens')
        }

@st.cache_resource
def load_model():
    """Load the trained model"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = CognitiveRadiologySystem(device=device)
    checkpoint_path = 'checkpoints/full_system_best_epoch6.pt'
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    
    return model, tokenizer, device

def preprocess_image(image):
    """Preprocess image for model input"""
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)

def main():
    st.set_page_config(
        page_title="Cognitive Radiology Report Generator",
        page_icon="🏥",
        layout="wide"
    )
    
    st.title("🏥 Cognitive Radiology Report Generator")
    st.markdown("### Upload a chest X-ray image to generate an automated medical report")
    
    # Sidebar
    with st.sidebar:
        st.header("ℹ️ About")
        st.markdown("""
        This AI system uses a three-module cognitive architecture:
        
        **Module 1: PRO-FA**
        - Hierarchical visual perception
        - Pixel, region, and organ-level features
        
        **Module 2: MIX-MLP**
        - Disease classification
        - CheXpert 14-label prediction
        
        **Module 3: RCTA**
        - Triangular cognitive attention
        - Report generation
        """)
        
        st.divider()
        st.markdown("**Model:** full_system_best_epoch6.pt")
    
    # Main content
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📤 Upload Image")
        uploaded_file = st.file_uploader(
            "Choose a chest X-ray image",
            type=['png', 'jpg', 'jpeg'],
            help="Upload a chest X-ray image in PNG or JPEG format"
        )
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert('RGB')
            st.image(image, caption="Uploaded X-ray", use_container_width=True)
    
    with col2:
        st.subheader("📋 Generated Report")
        
        if uploaded_file is not None:
            with st.spinner("Analyzing image and generating report..."):
                try:
                    # Load model
                    model, tokenizer, device = load_model()
                    
                    # Preprocess image
                    image_tensor = preprocess_image(image).to(device)
                    
                    # Generate report
                    with torch.no_grad():
                        output = model(image_tensor, tokenizer=tokenizer)
                    
                    # Decode generated text
                    gen_findings_tokens = output['generated_findings'][0]
                    gen_impression_tokens = output['generated_impression'][0]
                    
                    findings_text = tokenizer.decode(gen_findings_tokens, skip_special_tokens=True)
                    impression_text = tokenizer.decode(gen_impression_tokens, skip_special_tokens=True)
                    
                    # Display report
                    st.markdown("**Findings:**")
                    st.info(findings_text if findings_text.strip() else "No significant findings.")
                    
                    st.markdown("**Impression:**")
                    st.info(impression_text if impression_text.strip() else "No impression generated.")
                    
                    # Display disease probabilities
                    st.divider()
                    st.markdown("**🔬 Disease Classification Probabilities**")
                    
                    disease_probs = output['disease_probs'][0].cpu().numpy()
                    
                    # Create two columns for disease labels
                    dcol1, dcol2 = st.columns(2)
                    
                    for idx, (label, prob) in enumerate(zip(CHEXPERT_LABELS, disease_probs)):
                        col = dcol1 if idx < 7 else dcol2
                        
                        with col:
                            # Color code based on probability
                            if prob > 0.5:
                                st.metric(label, f"{prob:.1%}", delta="Positive", delta_color="inverse")
                            else:
                                st.metric(label, f"{prob:.1%}")
                    
                    st.success("✅ Report generated successfully!")
                    
                except Exception as e:
                    st.error(f"Error generating report: {str(e)}")
                    st.exception(e)
        else:
            st.info("👆 Upload an image to generate a report")
    
    # Footer
    st.divider()
    st.markdown("""
    <div style='text-align: center; color: gray;'>
    <small>⚠️ This is a research prototype. Reports should be reviewed by qualified medical professionals.</small>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
