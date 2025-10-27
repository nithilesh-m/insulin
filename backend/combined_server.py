#!/usr/bin/env python3
"""
Combined Backend Server

Sections:
- AUTH: registration, login, sessions, MongoDB
- MODEL: tokenizer/ProteinLM -> PCA -> MLP predictions
- OAUTH: Google OAuth login flow
"""

import os
import requests
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import joblib
from dotenv import load_dotenv
from flask import Flask, request, jsonify, session, redirect
from flask_cors import CORS
from pymongo import MongoClient
from pymongo.errors import DuplicateKeyError
import bcrypt
import logging
from datetime import datetime, timedelta
import secrets
import random
from twilio.rest import Client as TwilioClient
from difflib import SequenceMatcher
from scipy.spatial.distance import hamming
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import pearsonr

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', secrets.token_hex(32))
# Allow frontend origin with credentials
CORS(app, supports_credentials=True, resources={r"/*": {"origins": ["http://localhost:5173", "http://127.0.0.1:5173"]}})

# MongoDB configuration
MONGO_URI = os.environ.get('MONGO_URI', 'mongodb://localhost:27017/')
client = MongoClient(MONGO_URI)
db = client['protein_prediction_db']
users_collection = db['users']
otps_collection = db['otps']
try:
    users_collection.create_index('username', unique=True)
except Exception as e:
    logger.warning(f"Index creation warning: {e}")

# Session configuration
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(hours=24)

def hash_password(password):
    """Hash a password using bcrypt"""
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

def verify_password(password, hashed):
    """Verify a password against its hash"""
    return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))

# ==============
# OTP UTILITIES
# ==============

TWILIO_ACCOUNT_SID = os.environ.get('TWILIO_ACCOUNT_SID')
TWILIO_AUTH_TOKEN = os.environ.get('TWILIO_AUTH_TOKEN')
TWILIO_PHONE_NUMBER = os.environ.get('TWILIO_PHONE_NUMBER')
DEFAULT_COUNTRY_CODE = os.environ.get('DEFAULT_COUNTRY_CODE', '+91')
DEV_FAKE_OTP = os.environ.get('DEV_FAKE_OTP', 'false').lower() == 'true'

def normalize_phone(mobile: str) -> str:
    m = mobile.strip().replace(' ', '')
    if m.startswith('+'):
        return m
    if m.startswith('0'):
        m = m.lstrip('0')
    return f"{DEFAULT_COUNTRY_CODE}{m}"

def generate_otp(length: int = 6) -> str:
    return ''.join(str(random.randint(0, 9)) for _ in range(length))

def send_sms_otp(phone: str, code: str) -> bool:
    if DEV_FAKE_OTP or not (TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN and TWILIO_PHONE_NUMBER):
        logger.info(f"[DEV] OTP for {phone}: {code}")
        return True
    try:
        client = TwilioClient(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
        client.messages.create(
            body=f"Your verification code is {code}",
            from_=TWILIO_PHONE_NUMBER,
            to=phone,
        )
        return True
    except Exception as e:
        logger.error(f"Twilio send error: {e}")
        return False

# =============================
# MODEL SECTION: Inference setup
# =============================

# Config
MAX_TOKEN_LENGTH = 512
WINDOW_LEN = 50
HIDDEN_DIM_1 = 512
HIDDEN_DIM_2 = 256
DROPOUT = 0.4

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"[Model] Using device: {device}")

# Globals
custom_protein_lm = None
mlp_model = None
label_encoder = None
pca_model = None
sequence_generator = None
protein_to_smile = None

# Custom tokenizer setup (same as model_server)
AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")
SPECIAL_TOKENS = ["<PAD>", "<SOS>", "<EOS>", "<UNK>"]
VOCAB = SPECIAL_TOKENS + AMINO_ACIDS
token2idx = {tok: idx for idx, tok in enumerate(VOCAB)}
PAD_ID = token2idx["<PAD>"]
SOS_ID = token2idx["<SOS>"]
EOS_ID = token2idx["<EOS>"]

def tokenize(seq: str):
    ids = [SOS_ID]
    for ch in seq.strip().upper():
        ids.append(token2idx.get(ch, token2idx["<UNK>"]))
    ids.append(EOS_ID)
    return ids

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
    def __init__(self, vocab_size=len(VOCAB), d_model=256, nhead=8, num_layers=6, d_ff=1024, max_len=1024, dropout=0.1):
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
    global custom_protein_lm, mlp_model, label_encoder, pca_model, sequence_generator, protein_to_smile
    try:
        logger.info("[Model] Loading models...")
        # ProteinLM (random if checkpoint missing)
        custom_protein_lm = ProteinLM()
        ckpt_path = "models/custom_protein_lm.pt"
        if os.path.exists(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location=device)
            custom_protein_lm.load_state_dict(ckpt.get("model_state", ckpt), strict=False)
        else:
            logger.warning("[Model] Custom checkpoint not found; using random init")
        custom_protein_lm.to(device).eval()

        # Label encoder, PCA
        label_encoder = joblib.load("models/label_encoder.pkl")
        pca_model = joblib.load("models/pca_model.pkl")
        pca_dim = getattr(pca_model, 'n_components_', None) or 512

        # MLP
        num_classes = len(label_encoder.classes_)
        mlp_model = MLP(input_dim=pca_dim, num_classes=num_classes).to(device)
        state = torch.load("models/best_mlp_medium_adv.pth", map_location=device)
        mlp_model.load_state_dict(state, strict=False)
        mlp_model.eval()
        
        # Sequence Generator
        try:
            sequence_generator = torch.load("models/Sequence_Generator.pt", map_location=device)
            sequence_generator.eval()
            logger.info("[Model] Sequence generator loaded")
        except Exception as e:
            logger.warning(f"[Model] Sequence generator not loaded: {e}")
            sequence_generator = None
        
        # Protein to SMILES Generator
        try:
            protein_to_smile = torch.load("models/Protein_to_Smile.pt", map_location=device)
            protein_to_smile.eval()
            logger.info("[Model] Protein to SMILES generator loaded")
        except Exception as e:
            logger.warning(f"[Model] Protein to SMILES generator not loaded: {e}")
            protein_to_smile = None
        
        logger.info("[Model] All models loaded")
    except Exception as e:
        logger.error(f"[Model] Load error: {e}")
        raise

@torch.no_grad()
def embed_sequence(seq: str):
    ids = tokenize(seq)
    if len(ids) > MAX_TOKEN_LENGTH:
        ids = ids[:MAX_TOKEN_LENGTH-1] + [EOS_ID]
    padded = ids + [PAD_ID] * (MAX_TOKEN_LENGTH - len(ids))
    input_ids = torch.tensor([padded], dtype=torch.long).to(device)
    mask = [1 if tok != PAD_ID else 0 for tok in padded]
    attention_mask = torch.tensor([mask], dtype=torch.long).to(device)
    out = custom_protein_lm(input_ids, attention_mask=attention_mask)
    return out.mean(dim=1).detach().cpu().float().numpy()

def preprocess_sequence(seq: str) -> str:
    s = seq.strip().upper()
    valid = set("ACDEFGHIKLMNPQRSTVWY")
    if not s or not all(c in valid for c in s):
        raise ValueError("Sequence contains invalid amino acids")
    if len(s) < 10:
        raise ValueError("Sequence too short (minimum 10 amino acids)")
    if len(s) > WINDOW_LEN:
        s = s[:WINDOW_LEN]
    else:
        s = s.ljust(WINDOW_LEN, 'A')
    return s

# Similarity calculation functions
def levenshtein_similarity(seq1: str, seq2: str) -> float:
    """Calculate Levenshtein similarity between two sequences"""
    return SequenceMatcher(None, seq1, seq2).ratio()

def hamming_similarity(seq1: str, seq2: str) -> float:
    """Calculate Hamming similarity between two sequences"""
    if len(seq1) != len(seq2):
        # Pad shorter sequence
        max_len = max(len(seq1), len(seq2))
        seq1 = seq1.ljust(max_len, 'X')
        seq2 = seq2.ljust(max_len, 'X')
    
    # Convert to numeric arrays for hamming distance
    seq1_nums = [ord(c) for c in seq1]
    seq2_nums = [ord(c) for c in seq2]
    
    hamming_dist = hamming(seq1_nums, seq2_nums)
    return 1 - hamming_dist  # Convert distance to similarity

def cosine_similarity_sequences(seq1: str, seq2: str) -> float:
    """Calculate cosine similarity between two sequences"""
    # Create one-hot encoding
    amino_acids = "ACDEFGHIKLMNPQRSTVWY"
    
    def encode_sequence(seq):
        encoding = []
        for char in seq:
            vector = [0] * len(amino_acids)
            if char in amino_acids:
                vector[amino_acids.index(char)] = 1
            encoding.append(vector)
        return np.array(encoding).flatten()
    
    vec1 = encode_sequence(seq1)
    vec2 = encode_sequence(seq2)
    
    # Ensure same length
    min_len = min(len(vec1), len(vec2))
    vec1 = vec1[:min_len]
    vec2 = vec2[:min_len]
    
    if np.linalg.norm(vec1) == 0 or np.linalg.norm(vec2) == 0:
        return 0.0
    
    return cosine_similarity([vec1], [vec2])[0][0]

def pearson_similarity(seq1: str, seq2: str) -> float:
    """Calculate Pearson correlation similarity between two sequences"""
    # Convert to numeric representation
    amino_acids = "ACDEFGHIKLMNPQRSTVWY"
    
    def encode_sequence(seq):
        return [amino_acids.index(c) if c in amino_acids else 0 for c in seq]
    
    vec1 = encode_sequence(seq1)
    vec2 = encode_sequence(seq2)
    
    # Ensure same length
    min_len = min(len(vec1), len(vec2))
    vec1 = vec1[:min_len]
    vec2 = vec2[:min_len]
    
    if len(vec1) < 2:
        return 0.0
    
    try:
        correlation, _ = pearsonr(vec1, vec2)
        return max(0, correlation)  # Return only positive correlations
    except:
        return 0.0

def generate_sequences(input_seq: str, num_sequences: int = 10) -> list:
    """Generate new protein sequences using the sequence generator model"""
    if sequence_generator is None:
        # Fallback: generate random sequences based on input
        generated = []
        for _ in range(num_sequences):
            # Simple mutation-based generation
            seq = list(input_seq)
            for i in range(random.randint(1, 3)):
                if len(seq) > 0:
                    pos = random.randint(0, len(seq) - 1)
                    seq[pos] = random.choice(AMINO_ACIDS)
            generated.append(''.join(seq))
        return generated
    
    try:
        # Tokenize input sequence
        input_ids = tokenize(input_seq)
        if len(input_ids) > MAX_TOKEN_LENGTH:
            input_ids = input_ids[:MAX_TOKEN_LEN-1] + [EOS_ID]
        
        # Pad sequence
        padded = input_ids + [PAD_ID] * (MAX_TOKEN_LENGTH - len(input_ids))
        input_tensor = torch.tensor([padded], dtype=torch.long).to(device)
        
        generated_sequences = []
        for _ in range(num_sequences):
            # Generate sequence (simplified - you may need to adjust based on your model)
            with torch.no_grad():
                # This is a placeholder - adjust based on your actual model architecture
                output = sequence_generator(input_tensor)
                # Convert output back to sequence
                # This part needs to be adapted based on your model's output format
                generated_seq = ''.join([AMINO_ACIDS[i % len(AMINO_ACIDS)] for i in range(len(input_seq))])
                generated_sequences.append(generated_seq)
        
        return generated_sequences
    except Exception as e:
        logger.error(f"Sequence generation error: {e}")
        # Fallback to random generation
        generated = []
        for _ in range(num_sequences):
            seq = list(input_seq)
            for i in range(random.randint(1, 3)):
                if len(seq) > 0:
                    pos = random.randint(0, len(seq) - 1)
                    seq[pos] = random.choice(AMINO_ACIDS)
            generated.append(''.join(seq))
        return generated

def calculate_sequence_probabilities(input_seq: str, generated_seqs: list) -> list:
    """Calculate similarity probabilities for generated sequences"""
    results = []
    
    for seq in generated_seqs:
        # Calculate all similarity metrics
        lev_sim = levenshtein_similarity(input_seq, seq)
        ham_sim = hamming_similarity(input_seq, seq)
        cos_sim = cosine_similarity_sequences(input_seq, seq)
        pear_sim = pearson_similarity(input_seq, seq)
        
        # Calculate average probability
        avg_prob = (lev_sim + ham_sim + cos_sim + pear_sim) / 4
        
        results.append({
            'sequence': seq,
            'average_probability': avg_prob,
            'levenshtein': lev_sim,
            'hamming': ham_sim,
            'cosine': cos_sim,
            'pearson': pear_sim
        })
    
    # Sort by average probability (descending)
    results.sort(key=lambda x: x['average_probability'], reverse=True)
    
    return results[:5]  # Return top 5

def generate_smiles(input_seq: str) -> str:
    """Generate SMILES structure from protein sequence"""
    if protein_to_smile is None:
        # Fallback: return a placeholder SMILES
        return "C[C@H](N)C(=O)O"  # Simple amino acid SMILES
    
    try:
        # Tokenize input sequence
        input_ids = tokenize(input_seq)
        if len(input_ids) > MAX_TOKEN_LENGTH:
            input_ids = input_ids[:MAX_TOKEN_LENGTH-1] + [EOS_ID]
        
        # Pad sequence
        padded = input_ids + [PAD_ID] * (MAX_TOKEN_LENGTH - len(input_ids))
        input_tensor = torch.tensor([padded], dtype=torch.long).to(device)
        
        # Generate SMILES (simplified - adjust based on your model architecture)
        with torch.no_grad():
            # This is a placeholder - adjust based on your actual model architecture
            output = protein_to_smile(input_tensor)
            # Convert output to SMILES string
            # This part needs to be adapted based on your model's output format
            smiles = "C[C@H](N)C(=O)O"  # Placeholder SMILES
        
        return smiles
    except Exception as e:
        logger.error(f"SMILES generation error: {e}")
        # Fallback to simple SMILES
        return "C[C@H](N)C(=O)O"

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'healthy',
        'service': 'authentication+prediction',
        'models_loaded': all([custom_protein_lm is not None, mlp_model is not None, label_encoder is not None, pca_model is not None])
    })

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json() or {}
        seq = data.get('sequence', '')
        proc = preprocess_sequence(seq)
        emb = embed_sequence(proc)
        if pca_model is not None:
            emb = pca_model.transform(emb)
        X = torch.tensor(emb, dtype=torch.float32).to(device)
        logits = mlp_model(X)
        probs = F.softmax(logits, dim=1)
        idx = torch.argmax(probs, dim=1).item()
        conf = probs[0, idx].item()
        pred = label_encoder.classes_[idx]
        prob_map = {label_encoder.classes_[i]: probs[0, i].item() for i in range(len(label_encoder.classes_))}
        return jsonify({
            'success': True,
            'result': {
                'prediction': pred,
                'confidence': conf,
                'probabilities': prob_map
            },
            'processed_sequence': proc
        })
    except ValueError as ve:
        return jsonify({'success': False, 'error': str(ve)}), 400
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({'success': False, 'error': 'Prediction failed'}), 500

@app.route('/classes', methods=['GET'])
def classes():
    if label_encoder is None:
        return jsonify({'success': False, 'error': 'Model not loaded'}), 500
    return jsonify({'success': True, 'classes': label_encoder.classes_.tolist()})

@app.route('/generate-sequences', methods=['POST'])
def generate_sequences_endpoint():
    """Generate new protein sequences and calculate similarity probabilities"""
    try:
        data = request.get_json() or {}
        input_sequence = data.get('sequence', '').strip().upper()
        
        # Validation
        if not input_sequence:
            return jsonify({'success': False, 'error': 'Sequence is required'}), 400
        
        valid_amino_acids = set("ACDEFGHIKLMNPQRSTVWY")
        if not all(c in valid_amino_acids for c in input_sequence):
            return jsonify({'success': False, 'error': 'Invalid amino acid sequence'}), 400
        
        if len(input_sequence) < 10:
            return jsonify({'success': False, 'error': 'Sequence too short (minimum 10 amino acids)'}), 400
        
        # Generate sequences
        generated_seqs = generate_sequences(input_sequence, num_sequences=10)
        
        # Calculate probabilities
        results = calculate_sequence_probabilities(input_sequence, generated_seqs)
        
        return jsonify({
            'success': True,
            'input_sequence': input_sequence,
            'generated_sequences': results
        })
        
    except Exception as e:
        logger.error(f"Sequence generation error: {e}")
        return jsonify({'success': False, 'error': 'Sequence generation failed'}), 500

@app.route('/generate-smiles', methods=['POST'])
def generate_smiles_endpoint():
    """Generate SMILES structure from protein sequence"""
    try:
        data = request.get_json() or {}
        input_sequence = data.get('sequence', '').strip().upper()
        
        # Validation
        if not input_sequence:
            return jsonify({'success': False, 'error': 'Sequence is required'}), 400
        
        valid_amino_acids = set("ACDEFGHIKLMNPQRSTVWY")
        if not all(c in valid_amino_acids for c in input_sequence):
            return jsonify({'success': False, 'error': 'Invalid amino acid sequence'}), 400
        
        if len(input_sequence) < 10:
            return jsonify({'success': False, 'error': 'Sequence too short (minimum 10 amino acids)'}), 400
        
        # Generate SMILES
        smiles = generate_smiles(input_sequence)
        
        return jsonify({
            'success': True,
            'input_sequence': input_sequence,
            'smiles': smiles
        })
        
    except Exception as e:
        logger.error(f"SMILES generation error: {e}")
        return jsonify({'success': False, 'error': 'SMILES generation failed'}), 500

# ======================
# AUTH ENDPOINTS
# ======================

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'authentication',
        'timestamp': datetime.utcnow().isoformat()
    })

@app.route('/api/register', methods=['POST'])
def register():
    """Register a new user"""
    try:
        data = request.get_json()
        username = data.get('username', '').strip()
        password = data.get('password', '').strip()
        
        # Validation
        if not username or not password:
            return jsonify({
                'success': False,
                'error': 'Username and password are required'
            }), 400
        
        if len(username) < 3:
            return jsonify({
                'success': False,
                'error': 'Username must be at least 3 characters long'
            }), 400
        
        if len(password) < 6:
            return jsonify({
                'success': False,
                'error': 'Password must be at least 6 characters long'
            }), 400
        
        # Check if user already exists
        existing_user = users_collection.find_one({'username': username})
        if existing_user:
            return jsonify({
                'success': False,
                'error': 'Username already exists'
            }), 400
        
        # Create new user
        hashed_password = hash_password(password)
        user_data = {
            'username': username,
            'password': hashed_password,
            'created_at': datetime.utcnow(),
            'last_login': None
        }
        
        try:
            result = users_collection.insert_one(user_data)
        except DuplicateKeyError:
            return jsonify({'success': False, 'error': 'Username already exists'}), 400
        
        # Create session
        session['user_id'] = str(result.inserted_id)
        session['username'] = username
        session.permanent = True
        
        logger.info(f"New user registered: {username}")
        
        return jsonify({
            'success': True,
            'message': 'User registered successfully',
            'user': {
                'id': str(result.inserted_id),
                'username': username
            }
        })
        
    except Exception as e:
        logger.error(f"Registration error: {str(e)}")
        return jsonify({
            'success': False,
            'error': 'Registration failed'
        }), 500

@app.route('/api/login', methods=['POST'])
def login():
    """Login user"""
    try:
        data = request.get_json()
        username = data.get('username', '').strip()
        password = data.get('password', '').strip()
        
        # Validation
        if not username or not password:
            return jsonify({
                'success': False,
                'error': 'Username and password are required'
            }), 400
        
        # Find user by username or mobile (user can enter either)
        user = users_collection.find_one({'$or': [
            {'username': username},
            {'mobile': username}
        ]})
        if not user:
            return jsonify({
                'success': False,
                'error': 'Invalid username or password'
            }), 401
        
        # Verify password
        if not verify_password(password, user['password']):
            return jsonify({
                'success': False,
                'error': 'Invalid username or password'
            }), 401
        
        # Update last login
        users_collection.update_one(
            {'_id': user['_id']},
            {'$set': {'last_login': datetime.utcnow()}}
        )
        
        # Create session
        session['user_id'] = str(user['_id'])
        session['username'] = username
        session.permanent = True
        
        logger.info(f"User logged in: {username}")
        
        return jsonify({
            'success': True,
            'message': 'Login successful',
            'user': {
                'id': str(user['_id']),
                'username': username
            }
        })
        
    except Exception as e:
        logger.error(f"Login error: {str(e)}")
        return jsonify({
            'success': False,
            'message': 'Login failed'
        }), 500

@app.route('/api/request-otp-mobile', methods=['POST'])
def request_otp_mobile():
    try:
        data = request.get_json() or {}
        mobile = data.get('mobile', '').strip()
        username = data.get('username', '').strip()
        if not mobile or not username:
            return jsonify({'success': False, 'error': 'Username and mobile are required'}), 400
        phone = normalize_phone(mobile)
        code = generate_otp()
        expires = datetime.utcnow() + timedelta(minutes=10)
        otps_collection.delete_many({'phone': phone})
        otps_collection.insert_one({'phone': phone, 'code': code, 'expires_at': expires, 'verified': False, 'username': username})
        if not send_sms_otp(phone, code):
            return jsonify({'success': False, 'error': 'Failed to send OTP'}), 500
        return jsonify({'success': True})
    except Exception as e:
        logger.error(f"request_otp_mobile error: {e}")
        return jsonify({'success': False, 'error': 'request_otp_mobile_failed'}), 500

@app.route('/api/verify-otp-mobile', methods=['POST'])
def verify_otp_mobile():
    try:
        data = request.get_json() or {}
        mobile = data.get('mobile', '').strip()
        otp = data.get('otp', '').strip()
        phone = normalize_phone(mobile)
        rec = otps_collection.find_one({'phone': phone})
        if not rec:
            return jsonify({'success': False, 'error': 'otp_not_found'}), 400
        if rec['expires_at'] < datetime.utcnow():
            return jsonify({'success': False, 'error': 'otp_expired'}), 400
        if rec['code'] != otp:
            return jsonify({'success': False, 'error': 'otp_invalid'}), 400
        otps_collection.update_one({'_id': rec['_id']}, {'$set': {'verified': True}})
        # Keep info in session for password setup
        session['pending_mobile'] = {'phone': phone, 'username': rec.get('username', phone)}
        return jsonify({'success': True})
    except Exception as e:
        logger.error(f"verify_otp_mobile error: {e}")
        return jsonify({'success': False, 'error': 'verify_otp_mobile_failed'}), 500

@app.route('/api/set-password-mobile', methods=['POST'])
def set_password_mobile():
    try:
        data = request.get_json() or {}
        password = data.get('password', '').strip()
        pend = session.get('pending_mobile')
        if not pend:
            return jsonify({'success': False, 'error': 'no_pending_mobile'}), 400
        if not password or len(password) < 6:
            return jsonify({'success': False, 'error': 'weak_password'}), 400
        username = pend['username']
        phone = pend['phone']
        # Create or update user
        existing = users_collection.find_one({'username': username})
        if existing:
            users_collection.update_one({'_id': existing['_id']}, {'$set': {'password': hash_password(password), 'mobile': phone}})
            user_id = str(existing['_id'])
        else:
            result = users_collection.insert_one({
                'username': username,
                'password': hash_password(password),
                'mobile': phone,
                'created_at': datetime.utcnow(),
                'last_login': datetime.utcnow(),
            })
            user_id = str(result.inserted_id)
        session.pop('pending_mobile', None)
        session['user_id'] = user_id
        session['username'] = username
        session.permanent = True
        return jsonify({'success': True, 'user': {'id': user_id, 'username': username}})
    except Exception as e:
        logger.error(f"set_password_mobile error: {e}")
        return jsonify({'success': False, 'error': 'set_password_mobile_failed'}), 500

@app.route('/api/logout', methods=['POST'])
def logout():
    """Logout user"""
    try:
        username = session.get('username', 'Unknown')
        session.clear()
        
        logger.info(f"User logged out: {username}")
        
        return jsonify({
            'success': True,
            'message': 'Logout successful'
        })
        
    except Exception as e:
        logger.error(f"Logout error: {str(e)}")
        return jsonify({
            'success': False,
            'message': 'Logout failed'
        }), 500

@app.route('/api/check-auth', methods=['GET'])
def check_auth():
    """Check if user is authenticated"""
    try:
        if 'user_id' in session and 'username' in session:
            return jsonify({
                'success': True,
                'authenticated': True,
                'user': {
                    'id': session['user_id'],
                    'username': session['username']
                }
            })
        else:
            return jsonify({
                'success': True,
                'authenticated': False
            })
            
    except Exception as e:
        logger.error(f"Auth check error: {str(e)}")
        return jsonify({
            'success': False,
            'message': 'Auth check failed'
        }), 500

# ======================
# OAUTH SECTION: Google
# ======================
@app.route('/api/auth/google', methods=['GET'])
def google_auth():
    """Initiate Google OAuth flow - redirect to Google login page"""
    try:
        # Get Google OAuth credentials from environment variables
        GOOGLE_CLIENT_ID = os.environ.get('GOOGLE_CLIENT_ID')
        
        if not GOOGLE_CLIENT_ID:
            return jsonify({'error': 'Google OAuth not configured. Please set GOOGLE_CLIENT_ID environment variable.'}), 500
        
        # Google OAuth authorization URL
        auth_url = 'https://accounts.google.com/o/oauth2/v2/auth'
        params = {
            'client_id': GOOGLE_CLIENT_ID,
            'redirect_uri': 'http://localhost:5001/api/auth/google/callback',
            'scope': 'openid email profile',
            'response_type': 'code',
            'access_type': 'offline',
            'prompt': 'select_account'
        }
        
        # Build the authorization URL with proper URL encoding
        from urllib.parse import urlencode
        auth_url_with_params = f"{auth_url}?{urlencode(params)}"
        
        return redirect(auth_url_with_params)
        
    except Exception as e:
        logger.error(f"Google OAuth error: {str(e)}")
        return jsonify({'error': 'Google OAuth failed'}), 500

@app.route('/api/auth/google/callback', methods=['GET'])
def google_auth_callback():
    """Handle Google OAuth callback"""
    try:
        code = request.args.get('code')
        error = request.args.get('error')
        
        if error:
            logger.error(f"Google OAuth error: {error}")
            # Redirect to frontend with error
            return redirect(f"http://localhost:5173?auth=error&error={error}")
        
        if not code:
            return redirect("http://localhost:5173?auth=error&error=no_code")
        
        # Exchange code for tokens
        try:
            # Get Google OAuth credentials
            GOOGLE_CLIENT_ID = os.environ.get('GOOGLE_CLIENT_ID')
            GOOGLE_CLIENT_SECRET = os.environ.get('GOOGLE_CLIENT_SECRET')
            
            if not GOOGLE_CLIENT_ID or not GOOGLE_CLIENT_SECRET:
                return redirect("http://localhost:5173?auth=error&error=oauth_not_configured")
            
            token_url = 'https://oauth2.googleapis.com/token'
            token_data = {
                'client_id': GOOGLE_CLIENT_ID,
                'client_secret': GOOGLE_CLIENT_SECRET,
                'code': code,
                'grant_type': 'authorization_code',
                'redirect_uri': 'http://localhost:5001/api/auth/google/callback'
            }
            
            token_response = requests.post(token_url, data=token_data)
            token_response.raise_for_status()
            tokens = token_response.json()
            
            # Get user info
            user_info_url = 'https://www.googleapis.com/oauth2/v2/userinfo'
            headers = {'Authorization': f"Bearer {tokens['access_token']}"}
            user_response = requests.get(user_info_url, headers=headers)
            user_response.raise_for_status()
            user_info = user_response.json()
            
            # Check if user exists or create new user
            google_id = user_info.get('id')
            email = user_info.get('email')
            name = user_info.get('name')
            
            if not google_id or not email:
                return redirect("http://localhost:5173?auth=error&error=invalid_user_info")
            
            # Check if user exists by Google ID
            existing_user = users_collection.find_one({'google_id': google_id})
            
            if existing_user:
                # Update last login
                users_collection.update_one(
                    {'_id': existing_user['_id']},
                    {'$set': {'last_login': datetime.utcnow()}}
                )
                user_id = str(existing_user['_id'])
                username = existing_user.get('username', email.split('@')[0])
            else:
                # Create new user
                user_data = {
                    'google_id': google_id,
                    'email': email,
                    'username': email.split('@')[0],
                    'name': name,
                    'created_at': datetime.utcnow(),
                    'last_login': datetime.utcnow()
                }
                
                result = users_collection.insert_one(user_data)
                user_id = str(result.inserted_id)
                username = email.split('@')[0]
            
            # Create session
            session['user_id'] = user_id
            session['username'] = username
            session.permanent = True
            
            logger.info(f"Google OAuth login successful: {username}")
            # Redirect to frontend with success
            return redirect(f"http://localhost:5173?auth=success&user={username}")
            
        except requests.RequestException as e:
            logger.error(f"Google OAuth token exchange error: {str(e)}")
            return redirect("http://localhost:5173?auth=error&error=token_exchange_failed")
        
    except Exception as e:
        logger.error(f"Google OAuth callback error: {str(e)}")
        return redirect("http://localhost:5173?auth=error&error=callback_failed")

@app.route('/api/set-password-google', methods=['POST'])
def set_password_google():
    """Set a password for the currently authenticated Google user"""
    try:
        if 'user_id' not in session:
            return jsonify({'success': False, 'error': 'not_authenticated'}), 401
        data = request.get_json() or {}
        password = (data.get('password') or '').strip()
        if len(password) < 6:
            return jsonify({'success': False, 'error': 'weak_password'}), 400
        user_id = session['user_id']
        users_collection.update_one({'_id': ObjectId(user_id)}, {
            '$set': {
                'password': hash_password(password),
                'last_login': datetime.utcnow()
            }
        })
        return jsonify({'success': True})
    except Exception as e:
        logger.error(f"set_password_google error: {e}")
        return jsonify({'success': False, 'error': 'set_password_google_failed'}), 500

if __name__ == '__main__':
    logger.info("Starting Authentication Server...")
    logger.info(f"MongoDB URI: {MONGO_URI}")
    # Load models at startup
    try:
        load_models()
    except Exception as e:
        logger.error(f"Failed to load models at startup: {e}")
    app.run(host='0.0.0.0', port=5001, debug=True)
