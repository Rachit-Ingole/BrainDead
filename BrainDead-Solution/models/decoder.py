import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict


class RCTA(nn.Module):

    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 8,
        vocab_size: int = 5000,
        max_seq_len: int = 256,
        num_decoder_layers: int = 2,
        dropout: float = 0.1
    ):

        super().__init__()
        
        self.embed_dim = embed_dim
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        

        self.clinical_vocab_size = 1000  # Small vocabulary for clinical indications
        self.clinical_embedding = nn.Embedding(self.clinical_vocab_size, embed_dim)
        self.clinical_encoder = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            batch_first=True
        )

        self.disease_projection = nn.Sequential(
            nn.Linear(14, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, embed_dim)
        )
        

        self.attention_image_to_clinical = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Step 2: Context attends to Disease embedding
        # Query: context_vector, Key/Value: disease_embedding
        self.attention_context_to_disease = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Step 3: Hypothesis attends back to Image features
        # Query: hypothesis_vector, Key/Value: region_features
        self.attention_hypothesis_to_image = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.norm3 = nn.LayerNorm(embed_dim)

        self.fusion = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.positional_encoding = nn.Parameter(
            torch.randn(1, max_seq_len, embed_dim)
        )

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=num_decoder_layers
        )

        self.findings_head = nn.Linear(embed_dim, vocab_size)
        self.impression_head = nn.Linear(embed_dim, vocab_size)

        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights for better convergence"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def encode_clinical_text(self, clinical_text_ids: torch.Tensor) -> torch.Tensor:
        """
        Encode clinical indication text
        
        Args:
            clinical_text_ids: (B, L) - tokenized clinical text
            
        Returns:
            clinical_embedding: (B, 1, 256) - pooled clinical context
        """
        # Embed tokens
        embedded = self.clinical_embedding(clinical_text_ids)  # (B, L, 256)
        
        # Encode with transformer
        encoded = self.clinical_encoder(embedded)  # (B, L, 256)
        
        # Mean pooling to get single vector per batch
        clinical_embedding = encoded.mean(dim=1, keepdim=True)  # (B, 1, 256)
        
        return clinical_embedding
    
    def triangular_attention(
        self,
        region_features: torch.Tensor,
        organ_feature: torch.Tensor,
        disease_logits: torch.Tensor,
        clinical_text_ids: Optional[torch.Tensor] = None
    ) -> torch.Tensor:

        B = region_features.size(0)

        if clinical_text_ids is not None:
            clinical_embedding = self.encode_clinical_text(clinical_text_ids)
        else:
            clinical_embedding = torch.zeros(B, 1, self.embed_dim, device=region_features.device)

        context_vector, _ = self.attention_image_to_clinical(
            query=region_features,
            key=clinical_embedding,
            value=clinical_embedding
        )
        context_vector = context_vector.mean(dim=1, keepdim=True)
        context_vector = self.norm1(context_vector)

        disease_embedding = self.disease_projection(disease_logits)
        disease_embedding = disease_embedding.unsqueeze(1)

        hypothesis_vector, _ = self.attention_context_to_disease(
            query=context_vector,
            key=disease_embedding,
            value=disease_embedding
        )
        hypothesis_vector = self.norm2(hypothesis_vector)

        verified_vector, attention_weights = self.attention_hypothesis_to_image(
            query=hypothesis_vector,
            key=region_features,
            value=region_features         # (B, R, 256)
        )  # Output: (B, 1, 256)
        
        verified_vector = self.norm3(verified_vector)
        verified_vector = verified_vector.squeeze(1)  # (B, 256)
        
        # Combine verified representation with global organ feature
        combined = torch.cat([verified_vector, organ_feature], dim=-1)  # (B, 512)
        verified_representation = self.fusion(combined)  # (B, 256)
        
        return verified_representation
    
    def decode_report(
        self,
        memory: torch.Tensor,
        target_tokens: torch.Tensor,
        report_type: str = "findings"
    ) -> torch.Tensor:
        """
        Decode report text using transformer decoder
        
        Args:
            memory: (B, 256) - verified representation from triangular attention
            target_tokens: (B, T) - target token sequence
            report_type: "findings" or "impression"
            
        Returns:
            logits: (B, T, vocab_size)
        """
        B, T = target_tokens.size()
        
        # Embed target tokens
        token_emb = self.token_embedding(target_tokens)  # (B, T, 256)
        
        # Add positional encoding
        pos_enc = self.positional_encoding[:, :T, :]  # (1, T, 256)
        decoder_input = token_emb + pos_enc  # (B, T, 256)
        
        # Expand memory to sequence format for cross-attention
        memory = memory.unsqueeze(1)  # (B, 1, 256)
        
        # Generate causal mask for autoregressive decoding
        causal_mask = nn.Transformer.generate_square_subsequent_mask(T).to(target_tokens.device)
        
        # Transformer decoder
        decoded = self.transformer_decoder(
            tgt=decoder_input,           # (B, T, 256)
            memory=memory,               # (B, 1, 256)
            tgt_mask=causal_mask        # (T, T)
        )  # Output: (B, T, 256)
        
        # Project to vocabulary
        if report_type == "findings":
            logits = self.findings_head(decoded)  # (B, T, vocab_size)
        else:  # impression
            logits = self.impression_head(decoded)  # (B, T, vocab_size)
        
        return logits
    
    def forward(
        self,
        region_features: torch.Tensor,
        organ_feature: torch.Tensor,
        disease_logits: torch.Tensor,
        findings_tokens: torch.Tensor,
        impression_tokens: torch.Tensor,
        clinical_text_ids: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass of RCTA module
        
        Args:
            region_features: (B, R, 256) - from PRO-FA Module 1
            organ_feature: (B, 256) - from PRO-FA Module 1
            disease_logits: (B, 14) - from MIX-MLP Module 2
            findings_tokens: (B, T1) - ground truth findings tokens
            impression_tokens: (B, T2) - ground truth impression tokens
            clinical_text_ids: (B, L) - optional clinical indication
            
        Returns:
            Dictionary containing:
                - findings_logits: (B, T1, vocab_size)
                - impression_logits: (B, T2, vocab_size)
                - verified_representation: (B, 256)
        """
        verified_representation = self.triangular_attention(
            region_features=region_features,
            organ_feature=organ_feature,
            disease_logits=disease_logits,
            clinical_text_ids=clinical_text_ids
        )  # (B, 256)
        
        # Generate Findings Section
        findings_logits = self.decode_report(
            memory=verified_representation,
            target_tokens=findings_tokens,
            report_type="findings"
        )  # (B, T1, vocab_size)
        
        # Generate Impression Section
        impression_logits = self.decode_report(
            memory=verified_representation,
            target_tokens=impression_tokens,
            report_type="impression"
        )  # (B, T2, vocab_size)
        
        return {
            "findings_logits": findings_logits,
            "impression_logits": impression_logits,
            "verified_representation": verified_representation
        }
    
    @torch.no_grad()
    def generate(
        self,
        region_features: torch.Tensor,
        organ_feature: torch.Tensor,
        disease_logits: torch.Tensor,
        clinical_text_ids: Optional[torch.Tensor] = None,
        max_length: int = 128,
        start_token_id: int = 1,
        end_token_id: int = 2,
        temperature: float = 1.0,
        top_p: float = 0.9
    ) -> Dict[str, torch.Tensor]:
        """
        Generate report text autoregressively during inference
        
        Args:
            region_features: (B, R, 256)
            organ_feature: (B, 256)
            disease_logits: (B, 14)
            clinical_text_ids: (B, L) - optional
            max_length: Maximum generation length
            start_token_id: BOS token ID
            end_token_id: EOS token ID
            
        Returns:
            Dictionary containing:
                - findings_tokens: (B, T1) - generated findings
                - impression_tokens: (B, T2) - generated impression
                - verified_representation: (B, 256)
        """
        self.eval()
        B = region_features.size(0)
        device = region_features.device
        
        # Get verified representation
        verified_representation = self.triangular_attention(
            region_features=region_features,
            organ_feature=organ_feature,
            disease_logits=disease_logits,
            clinical_text_ids=clinical_text_ids
        )
        
        # Generate findings
        findings_tokens = self._generate_sequence(
            memory=verified_representation,
            max_length=max_length,
            start_token_id=start_token_id,
            end_token_id=end_token_id,
            report_type="findings",
            temperature=temperature,
            top_p=top_p
        )
        
        # Generate impression
        impression_tokens = self._generate_sequence(
            memory=verified_representation,
            max_length=max_length,
            start_token_id=start_token_id,
            end_token_id=end_token_id,
            report_type="impression",
            temperature=temperature,
            top_p=top_p
        )
        
        return {
            "findings_tokens": findings_tokens,
            "impression_tokens": impression_tokens,
            "verified_representation": verified_representation
        }
    
    def _generate_sequence(
        self,
        memory: torch.Tensor,
        max_length: int,
        start_token_id: int,
        end_token_id: int,
        report_type: str,
        temperature: float = 1.0,
        top_p: float = 0.9
    ) -> torch.Tensor:
        """
        Autoregressively generate a single sequence
        
        Args:
            memory: (B, 256) - verified representation
            max_length: Maximum sequence length
            start_token_id: BOS token
            end_token_id: EOS token
            report_type: "findings" or "impression"
            
        Returns:
            generated_tokens: (B, T)
        """
        B = memory.size(0)
        device = memory.device
        
        # Start with BOS token
        generated = torch.full((B, 1), start_token_id, dtype=torch.long, device=device)
        
        for _ in range(max_length - 1):
            # Get logits for current sequence
            logits = self.decode_report(
                memory=memory,
                target_tokens=generated,
                report_type=report_type
            )  # (B, T, vocab_size)
            
            # Get next token prediction with nucleus sampling
            next_token_logits = logits[:, -1, :] / temperature  # (B, vocab_size)
            
            # Apply nucleus (top-p) sampling
            sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True, dim=-1)
            cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
            
            # Remove tokens with cumulative probability above threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
            sorted_indices_to_remove[:, 0] = False
            
            # Set logits to -inf for removed indices
            for b in range(B):
                indices_to_remove = sorted_indices[b, sorted_indices_to_remove[b]]
                next_token_logits[b, indices_to_remove] = float('-inf')
            
            # Sample from filtered distribution
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)  # (B, 1)
            
            # Append to sequence
            generated = torch.cat([generated, next_token], dim=1)  # (B, T+1)
            
            # Check if all sequences have generated EOS
            if (next_token == end_token_id).all():
                break
        
        return generated


# Example Usage (for testing, not part of training)
if __name__ == "__main__":
    # Create model
    model = RCTA(
        embed_dim=256,
        num_heads=8,
        vocab_size=5000,
        max_seq_len=256,
        num_decoder_layers=2,
        dropout=0.1
    )
    
    # Dummy inputs
    batch_size = 4
    num_regions = 49  # e.g., 7x7 grid
    
    region_features = torch.randn(batch_size, num_regions, 256)
    organ_feature = torch.randn(batch_size, 256)
    disease_logits = torch.randn(batch_size, 14)
    
    findings_tokens = torch.randint(0, 5000, (batch_size, 50))
    impression_tokens = torch.randint(0, 5000, (batch_size, 30))
    
    clinical_text_ids = torch.randint(0, 1000, (batch_size, 20))
    
    # Forward pass (training mode)
    outputs = model(
        region_features=region_features,
        organ_feature=organ_feature,
        disease_logits=disease_logits,
        findings_tokens=findings_tokens,
        impression_tokens=impression_tokens,
        clinical_text_ids=clinical_text_ids
    )
    
    print("Findings logits shape:", outputs["findings_logits"].shape)
    print("Impression logits shape:", outputs["impression_logits"].shape)
    print("Verified representation shape:", outputs["verified_representation"].shape)
    
    # Generation mode (inference)
    generated = model.generate(
        region_features=region_features,
        organ_feature=organ_feature,
        disease_logits=disease_logits,
        clinical_text_ids=clinical_text_ids,
        max_length=50
    )
    
    print("\nGenerated findings shape:", generated["findings_tokens"].shape)
    print("Generated impression shape:", generated["impression_tokens"].shape)