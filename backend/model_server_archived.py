#!/usr/bin/env python3
"""
Model Prediction Backend Server
Handles protein sequence prediction using custom trained model
Separated from authentication to avoid interference
"""

import os
import time
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import joblib
from flask import Flask, request, jsonify
from flask_cors import CORS
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
# Allow frontend to call with credentials (cookies)
CORS(app, supports_credentials=True, resources={r"/*": {"origins": ["http://localhost:5173"]}})

# Configuration from your custom model training
MAX_TOKEN_LENGTH = 512
WINDOW_LEN = 50
# Default; will be overridden by loaded PCA if available
PCA_DIM = 512
HIDDEN_DIM_1 = 512  # From your training
HIDDEN_DIM_2 = 256  # From your training
DROPOUT = 0.4
BATCH_EMBED = 1

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

# Global variables for models
custom_protein_lm = None
mlp_model = None
label_encoder = None
pca_model = None

# Custom tokenizer from your training
AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")
SPECIAL_TOKENS = ["<PAD>", "<SOS>", "<EOS>", "<UNK>"]
VOCAB = SPECIAL_TOKENS + AMINO_ACIDS
token2idx = {tok: idx for idx, tok in enumerate(VOCAB)}
idx2token = {idx: tok for tok, idx in token2idx.items()}
PAD_ID = token2idx["<PAD>"]
SOS_ID = token2idx["<SOS>"]
EOS_ID = token2idx["<EOS>"]
UNK_ID = token2idx["<UNK>"]
VOCAB_SIZE = len(VOCAB)

def tokenize(seq: str):
    """Tokenize protein sequence using custom tokenizer"""
    ids = [SOS_ID]
    for ch in seq.strip():
        ids.append(token2idx.get(ch, UNK_ID))
    ids.append(EOS_ID)
    return ids

# Custom ProteinLM architecture from your training
class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_head, attn_dropout=0.1):
        super().__init__()
        assert d_model % n_head == 0
        self.n_head = n_head
        self.d_head = d_model // n_head
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(attn_dropout)
    
    def forward(self, x, mask=None):
        B, T, C = x.size()
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(B, T, self.n_head, self.d_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.d_head).transpose(1, 2)
        att = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_head)
        causal_mask = torch.tril(torch.ones(T, T, device=x.device)).unsqueeze(0).unsqueeze(0)
        att = att.masked_fill(causal_mask == 0, float("-inf"))
        if mask is not None:
            mask2 = mask.unsqueeze(1).unsqueeze(2)
            att = att.masked_fill(mask2 == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.dropout(att)
        out = torch.matmul(att, v)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(out)

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
    
    def forward(self, x):
        return self.net(x)

class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_head, d_ff, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_head, attn_dropout=dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model, d_ff, dropout)
    
    def forward(self, x, mask):
        x = x + self.attn(self.ln1(x), mask)
        x = x + self.ff(self.ln2(x))
        return x

class ProteinLM(nn.Module):
    def __init__(self, vocab_size=VOCAB_SIZE, d_model=256, nhead=8, num_layers=6, d_ff=1024, max_len=1024, dropout=0.1):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.layers = nn.ModuleList([TransformerBlock(d_model, nhead, d_ff, dropout) for _ in range(num_layers)])
        self.ln_f = nn.LayerNorm(d_model)
        self.max_len = max_len
        self.d_model = d_model

    def forward(self, input_ids, attention_mask=None):
        B, T = input_ids.size()
        positions = torch.arange(0, T, device=input_ids.device).unsqueeze(0)
        x = self.token_emb(input_ids) + self.pos_emb(positions)
        for layer in self.layers:
            x = layer(x, attention_mask)
        x = self.ln_f(x)
        return x

class MLP(nn.Module):
    def __init__(self, input_dim, hidden1=HIDDEN_DIM_1, hidden2=HIDDEN_DIM_2, num_classes=4, dropout=DROPOUT):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.BatchNorm1d(hidden1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden1, hidden2),
            nn.BatchNorm1d(hidden2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden2, num_classes)
        )
    
    def forward(self, x):
        return self.net(x)

def load_models():
    """Load all trained models"""
    global custom_protein_lm, mlp_model, label_encoder, pca_model
    
    try:
        logger.info("Loading models...")
        
        # Load custom ProteinLM
        logger.info("Loading custom ProteinLM...")
        custom_protein_lm = ProteinLM()
        ckpt_path = "models/custom_protein_lm.pt"
        
        if os.path.exists(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location=device)
            custom_protein_lm.load_state_dict(ckpt["model_state"], strict=False)
            logger.info("Custom ProteinLM loaded successfully!")
        else:
            logger.warning(f"Custom checkpoint not found at {ckpt_path}")
            logger.warning("Using randomly initialized ProteinLM (predictions will be inaccurate)")
        
        custom_protein_lm.to(device)
        custom_protein_lm.eval()
        
        # Load label encoder
        logger.info("Loading label encoder...")
        label_encoder = joblib.load("models/label_encoder.pkl")
        
        # Load PCA model
        logger.info("Loading PCA model...")
        pca_model = joblib.load("models/pca_model.pkl")
        # Derive PCA output dimension dynamically if possible
        derived_pca_dim = getattr(pca_model, 'n_components_', None)
        if isinstance(derived_pca_dim, int) and derived_pca_dim > 0:
            logger.info(f"Detected PCA components: {derived_pca_dim}")
            pca_dim = derived_pca_dim
        else:
            logger.warning(f"Could not detect PCA components; falling back to configured PCA_DIM={PCA_DIM}")
            pca_dim = PCA_DIM
        
        # Load MLP model
        logger.info("Loading MLP model...")
        num_classes = len(label_encoder.classes_)
        mlp_model = MLP(input_dim=pca_dim, num_classes=num_classes).to(device)
        # Load weights with tolerance for minor mismatches
        state = torch.load("models/best_mlp_medium_adv.pth", map_location=device)
        load_result = mlp_model.load_state_dict(state, strict=False)
        try:
            missing = list(getattr(load_result, 'missing_keys', []))
            unexpected = list(getattr(load_result, 'unexpected_keys', []))
        except Exception:
            missing, unexpected = [], []
        if missing:
            logger.warning(f"Missing keys when loading MLP weights: {missing}")
        if unexpected:
            logger.warning(f"Unexpected keys when loading MLP weights: {unexpected}")
        mlp_model.eval()
        
        logger.info("All models loaded successfully!")
        
    except Exception as e:
        logger.error(f"Error loading models: {str(e)}")
        raise e

@torch.no_grad()
def embed_sequence(seq):
    """Generate embeddings for a single sequence using custom ProteinLM"""
    try:
        custom_protein_lm.eval()
        
        # Tokenize sequence
        ids = tokenize(seq)
        if len(ids) > MAX_TOKEN_LENGTH:
            ids = ids[:MAX_TOKEN_LENGTH-1] + [EOS_ID]
        
        # Pad to max_length
        padded_ids = ids + [PAD_ID] * (MAX_TOKEN_LENGTH - len(ids))
        input_ids = torch.tensor([padded_ids], dtype=torch.long).to(device)
        
        # Create attention mask
        mask = [1 if tok != PAD_ID else 0 for tok in padded_ids]
        attention_mask = torch.tensor([mask], dtype=torch.long).to(device)
        
        # Forward through custom model
        outputs = custom_protein_lm(input_ids, attention_mask=attention_mask)
        mean_emb = outputs.mean(dim=1)  # Mean pooling
        return mean_emb.detach().cpu().float().numpy()
        
    except Exception as e:
        logger.error(f"Error generating embeddings: {str(e)}")
        raise e

def preprocess_sequence(seq):
    """Preprocess sequence similar to training"""
    try:
        # Clean sequence
        seq = seq.strip().upper()
        
        # Validate amino acids
        valid_aa = set("ACDEFGHIKLMNPQRSTVWY")
        if not all(c in valid_aa for c in seq):
            raise ValueError("Sequence contains invalid amino acids")
        
        if len(seq) < 10:
            raise ValueError("Sequence too short (minimum 10 amino acids)")
        
        # Truncate or pad to window length
        if len(seq) > WINDOW_LEN:
            seq = seq[:WINDOW_LEN]
        else:
            seq = seq.ljust(WINDOW_LEN, 'A')  # Pad with 'A' if too short
        
        return seq
        
    except Exception as e:
        logger.error(f"Error preprocessing sequence: {str(e)}")
        raise e

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'model_prediction',
        'timestamp': datetime.utcnow().isoformat(),
        'models_loaded': all([custom_protein_lm is not None, mlp_model is not None, 
                             label_encoder is not None, pca_model is not None])
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Make prediction on protein sequence"""
    try:
        data = request.get_json()
        sequence = data.get('sequence', '').strip()
        
        if not sequence:
            return jsonify({
                'success': False,
                'error': 'Sequence is required'
            }), 400
        
        # Preprocess sequence
        processed_seq = preprocess_sequence(sequence)
        
        # Generate embeddings
        embeddings = embed_sequence(processed_seq)
        
        # Apply PCA
        if pca_model is not None:
            embeddings = pca_model.transform(embeddings)
        
        # Make prediction
        with torch.no_grad():
            X = torch.tensor(embeddings, dtype=torch.float32).to(device)
            logits = mlp_model(X)
            probabilities = F.softmax(logits, dim=1)
            prediction_idx = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][prediction_idx].item()
        
        # Get prediction details
        prediction = label_encoder.classes_[prediction_idx]
        prob_dict = {
            label_encoder.classes_[i]: probabilities[0][i].item() 
            for i in range(len(label_encoder.classes_))
        }
        
        return jsonify({
            'success': True,
            'result': {
                'prediction': prediction,
                'confidence': confidence,
                'probabilities': prob_dict
            },
            'processed_sequence': processed_seq
        })
        
    except ValueError as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return jsonify({
            'success': False,
            'error': 'Prediction failed'
        }), 500

@app.route('/classes', methods=['GET'])
def get_classes():
    """Get available prediction classes"""
    try:
        if label_encoder is None:
            return jsonify({
                'success': False,
                'error': 'Model not loaded'
            }), 500
        
        return jsonify({
            'success': True,
            'classes': label_encoder.classes_.tolist()
        })
        
    except Exception as e:
        logger.error(f"Error getting classes: {str(e)}")
        return jsonify({
            'success': False,
            'error': 'Failed to get classes'
        }), 500

if __name__ == '__main__':
    logger.info("Starting Model Prediction Server...")
    load_models()
    app.run(host='0.0.0.0', port=5000, debug=True)
