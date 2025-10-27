# --- SETUP INSTRUCTIONS ---
# 1. For Email OTP (Gmail):
#    - Enable 2-Step Verification on your Google account.
#    - Generate an App Password at https://myaccount.google.com/apppasswords.
#    - Set EMAIL_ADDRESS and EMAIL_PASSWORD as environment variables (see below).
# 2. For Mobile OTP (Twilio):
#    - Sign up at https://www.twilio.com/try-twilio.
#    - Get your Account SID, Auth Token, and Twilio phone number.
#    - Set TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, TWILIO_PHONE_NUMBER as environment variables.
# 3. For MongoDB:
#    - Make sure MongoDB is running locally or update MONGO_URI for Atlas.
# 4. Environment variables to set (in .env or your shell):
#    EMAIL_ADDRESS=your_gmail_address@gmail.com
#    EMAIL_PASSWORD=your_gmail_app_password
#    TWILIO_ACCOUNT_SID=your_twilio_account_sid
#    TWILIO_AUTH_TOKEN=your_twilio_auth_token
#    TWILIO_PHONE_NUMBER=your_twilio_phone_number
#    MONGO_URI=mongodb://localhost:27017/

import os
from dotenv import load_dotenv
import torch
import torch.nn as nn
import numpy as np
import joblib
from flask import Flask, request, jsonify, session, redirect
from flask_cors import CORS
from transformers import AutoTokenizer, AutoModelForCausalLM
import logging
from pymongo import MongoClient
import bcrypt
from datetime import datetime, timedelta
import secrets
from functools import wraps
import random
import re
import smtplib
from email.mime.text import MIMEText
from twilio.rest import Client

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()
app = Flask(__name__)
app.secret_key = secrets.token_hex(16)  # Generate a random secret key
CORS(app, supports_credentials=True)

# MongoDB setup
MONGO_URI = os.environ.get('MONGO_URI', 'mongodb://localhost:27017/')
DB_NAME = "protein_prediction_db"
COLLECTION_NAME = "users"
client = MongoClient(MONGO_URI)
db = client[DB_NAME]
users_collection = db[COLLECTION_NAME]

# In-memory user store and OTP store (for demo)
# users = {}  # {email: {password: ..., mobile: ...}}
otp_store = {}  # {email/mobile: otp}

# Twilio config
TWILIO_ACCOUNT_SID = os.environ.get('TWILIO_ACCOUNT_SID')
TWILIO_AUTH_TOKEN = os.environ.get('TWILIO_AUTH_TOKEN')
TWILIO_PHONE_NUMBER = os.environ.get('TWILIO_PHONE_NUMBER')
# Default country code used when users enter national numbers (without +CC). Use E.164 format like '+91', '+1', etc.
DEFAULT_COUNTRY_CODE = os.environ.get('DEFAULT_COUNTRY_CODE', '+91')
# For safety, default fake OTP fallback to disabled. Set to 'true' explicitly only for local dev.
DEV_FAKE_OTP = os.environ.get('DEV_FAKE_OTP', 'false').lower() == 'true'


def normalize_mobile_to_e164(mobile_raw: str) -> str | None:
    """Normalize the provided phone number to E.164. Returns None if invalid.

    Rules:
    - If the input starts with '+', assume it's already E.164; keep digits and '+' only and validate length (7-15 digits total after '+').
    - Else, strip all non-digits, then prepend DEFAULT_COUNTRY_CODE and validate.
    """
    if not mobile_raw or not isinstance(mobile_raw, str):
        return None
    # Keep only digits and plus
    cleaned = re.sub(r"[^\d\+]", "", mobile_raw)
    if cleaned.startswith('+'):
        # Validate E.164: + followed by 7 to 15 digits total length
        if re.match(r"^\+[1-9]\d{6,14}$", cleaned):
            return cleaned
        return None
    else:
        digits = re.sub(r"\D", "", cleaned)
        if not digits:
            return None
        e164 = f"{DEFAULT_COUNTRY_CODE}{digits}"
        if re.match(r"^\+[1-9]\d{6,14}$", e164):
            return e164
        return None

# Email config
EMAIL_HOST = "smtp.gmail.com"
EMAIL_PORT = 587
EMAIL_ADDRESS = os.environ.get("EMAIL_ADDRESS")  # Your Gmail address
EMAIL_PASSWORD = os.environ.get("EMAIL_PASSWORD")  # Your Gmail App Password

# --- EMAIL OTP SENDING ---
def send_email_otp(email, otp):
    msg = MIMEText(f"Your OTP code is: {otp}")
    msg["Subject"] = "Your OTP Code"
    msg["From"] = EMAIL_ADDRESS
    msg["To"] = email
    try:
        with smtplib.SMTP(EMAIL_HOST, EMAIL_PORT) as server:
            server.starttls()
            server.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
            server.sendmail(EMAIL_ADDRESS, [email], msg.as_string())
        return True
    except Exception as e:
        print(f"SMTP error: {e}")
        return False

# --- SMS OTP SENDING ---
def send_sms(mobile_e164: str, otp: str) -> tuple[bool, str | None]:
    """Send SMS via Twilio to an E.164 number. Returns (success, error_message).
    In DEV_FAKE_OTP mode, logs the OTP and returns success for local testing.
    """
    try:
        if not (TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN and TWILIO_PHONE_NUMBER):
            if DEV_FAKE_OTP:
                logger.warning(f"DEV_FAKE_OTP enabled. OTP for {mobile_e164}: {otp}")
                return True, None
            else:
                msg = 'Twilio credentials not set (TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, TWILIO_PHONE_NUMBER)'
                logger.error(msg)
                return False, msg
        client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
        message = client.messages.create(
            body=f'Your OTP code is: {otp}',
            from_=TWILIO_PHONE_NUMBER,
            to=mobile_e164
        )
        if message.sid:
            return True, None
        return False, 'Twilio did not return a message SID'
    except Exception as e:
        logger.error(f"SMS send error: {e}")
        return False, str(e)

# Configuration
MAX_TOKEN_LENGTH = 512
WINDOW_LEN = 50
PCA_DIM = 512
HIDDEN_DIM_1 = 512
HIDDEN_DIM_2 = 256
DROPOUT = 0.4
PROGEN_MODEL = "hugohrban/progen2-medium"
BATCH_EMBED = 1  # Single prediction

# Global variables for models
progen = None
tokenizer = None
mlp_model = None
label_encoder = None
pca_model = None
device = None

# Authentication decorator
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not session.get('user_id'):
            return jsonify({'error': 'Authentication required'}), 401
        return f(*args, **kwargs)
    return decorated_function

class MLP(nn.Module):
    def __init__(self, input_dim, hidden1=HIDDEN_DIM_1, hidden2=HIDDEN_DIM_2, num_classes=5, dropout=DROPOUT):
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
    global progen, tokenizer, mlp_model, label_encoder, pca_model, device
    
    try:
        logger.info("Loading models...")
        
        # Set device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")
        
        # Load tokenizer and ProGen2 model
        logger.info("Loading ProGen2 model...")
        tokenizer = AutoTokenizer.from_pretrained(PROGEN_MODEL, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token
        
        progen = AutoModelForCausalLM.from_pretrained(PROGEN_MODEL, trust_remote_code=True)
        progen.eval()
        if device.type == 'cuda':
            progen = progen.to(device).half()
        else:
            progen = progen.to(device)
        
        # Load label encoder
        logger.info("Loading label encoder...")
        label_encoder = joblib.load("models/label_encoder.pkl")
        
        # Load PCA model
        logger.info("Loading PCA model...")
        pca_model = joblib.load("models/pca_model.pkl")
        
        # Load MLP model
        logger.info("Loading MLP model...")
        num_classes = len(label_encoder.classes_)
        mlp_model = MLP(input_dim=PCA_DIM, num_classes=num_classes).to(device)
        mlp_model.load_state_dict(torch.load("models/best_mlp_medium_adv.pth", map_location=device))
        mlp_model.eval()
        
        logger.info("All models loaded successfully!")
        
    except Exception as e:
        logger.error(f"Error loading models: {str(e)}")
        raise e

@torch.no_grad()
def embed_sequence(seq):
    """Generate embeddings for a single sequence"""
    try:
        progen.eval()
        toks = tokenizer([seq], return_tensors="pt", padding=True, truncation=True, max_length=MAX_TOKEN_LENGTH)
        toks = {k: v.to(device) for k, v in toks.items()}
        outputs = progen.transformer(**toks)
        mean_emb = outputs.last_hidden_state.mean(dim=1)
        return mean_emb.detach().cpu().float().numpy()
    except Exception as e:
        logger.error(f"Error generating embeddings: {str(e)}")
        raise e

def preprocess_sequence(seq):
    """Preprocess sequence similar to training"""
    seq = seq.upper().strip()
    
    # Validate amino acid sequence
    valid_amino_acids = set('ACDEFGHIKLMNPQRSTVWY')
    if not all(c in valid_amino_acids for c in seq):
        raise ValueError("Invalid amino acid sequence. Please use single-letter amino acid codes (A-Z, excluding B, J, O, U, X, Z).")
    
    # Handle sequence length
    if len(seq) < WINDOW_LEN:
        seq = seq.ljust(WINDOW_LEN, 'A')  # Pad with 'A'
    elif len(seq) > WINDOW_LEN:
        # Center crop
        mid = len(seq) // 2
        start = mid - WINDOW_LEN // 2
        seq = seq[start:start + WINDOW_LEN]
    
    return seq

def predict_sequence(seq):
    """Make prediction for a single sequence"""
    try:
        # Preprocess sequence
        processed_seq = preprocess_sequence(seq)
        
        # Generate embeddings
        embeddings = embed_sequence(processed_seq)
        
        # Apply PCA
        embeddings = pca_model.transform(embeddings)
        
        # Convert to tensor
        emb_tensor = torch.tensor(embeddings, dtype=torch.float32).to(device)
        
        # Make prediction
        with torch.no_grad():
            logits = mlp_model(emb_tensor)
            probabilities = torch.softmax(logits, dim=1)
            pred_idx = torch.argmax(logits, dim=1).item()
            confidence = probabilities[0, pred_idx].item()
            
        # Get class name
        pred_label = label_encoder.inverse_transform([pred_idx])[0]
        
        # Get all class probabilities
        class_probs = {}
        for i, class_name in enumerate(label_encoder.classes_):
            class_probs[class_name] = float(probabilities[0, i].item())
        
        return {
            'prediction': pred_label,
            'confidence': float(confidence),
            'probabilities': class_probs
        }
        
    except Exception as e:
        logger.error(f"Error in prediction: {str(e)}")
        raise e

# REMOVE: MongoDB Configuration, Initialization, and Authentication routes
# MongoDB and user authentication logic removed for OTP-based system

# --- EMAIL OTP ENDPOINTS ---
@app.route('/api/request-otp-email', methods=['POST'])
def request_otp_email():
    data = request.get_json()
    email = data.get('email')
    if not email:
        return jsonify({'error': 'Email required'}), 400
    otp = str(random.randint(100000, 999999))
    otp_store[email] = otp
    session['pending_email'] = email
    if send_email_otp(email, otp):
        return jsonify({'success': True, 'message': 'OTP sent to email'})
    else:
        return jsonify({'error': 'Failed to send OTP'}), 500

@app.route('/api/verify-otp-email', methods=['POST'])
def verify_otp_email():
    data = request.get_json()
    email = data.get('email')
    otp = data.get('otp')
    if not email or not otp:
        return jsonify({'error': 'Email and OTP required'}), 400
    if otp_store.get(email) == otp:
        session['otp_verified_email'] = email
        otp_store.pop(email, None)
        return jsonify({'success': True, 'message': 'OTP verified'})
    return jsonify({'error': 'Invalid OTP'}), 400

# Update set_password to store user in MongoDB
@app.route('/api/set-password', methods=['POST'])
def set_password():
    data = request.get_json()
    password = data.get('password')
    email = session.get('otp_verified_email')
    if not email or not password:
        return jsonify({'error': 'OTP verification required and password required'}), 400
    users_collection.update_one(
        {'_id': email},
        {'$set': {'password': password, 'email': email}},
        upsert=True
    )
    session['user_id'] = email
    session['username'] = email
    session.pop('otp_verified_email', None)
    return jsonify({'success': True, 'message': 'Password set and user logged in', 'user': {'id': email, 'username': email}})

@app.route('/api/request-otp-mobile', methods=['POST'])
def request_otp_mobile():
    data = request.get_json()
    mobile = data.get('mobile', '')
    username = data.get('username', '')
    if not mobile or not username:
        return jsonify({'error': 'Mobile and username required'}), 400
    # Normalize to E.164
    mobile_e164 = normalize_mobile_to_e164(mobile)
    if not mobile_e164:
        return jsonify({'error': 'Invalid mobile number format. Use full number or include country code.'}), 400
    otp = str(random.randint(100000, 999999))
    otp_store[mobile_e164] = otp
    session['pending_mobile'] = mobile_e164
    session['pending_username'] = username
    success, err = send_sms(mobile_e164, otp)
    if success:
        return jsonify({'success': True, 'message': 'OTP sent to mobile'})
    else:
        return jsonify({'error': f'Failed to send OTP: {err}'}), 500

@app.route('/api/verify-otp-mobile', methods=['POST'])
def verify_otp_mobile():
    data = request.get_json()
    mobile = data.get('mobile', '')
    otp = data.get('otp')
    if not mobile or not otp:
        return jsonify({'error': 'Mobile and OTP required'}), 400
    mobile_e164 = normalize_mobile_to_e164(mobile)
    if not mobile_e164:
        return jsonify({'error': 'Invalid mobile number format'}), 400
    if otp_store.get(mobile_e164) == otp:
        session['otp_verified_mobile'] = mobile_e164
        session['otp_verified_username'] = session.get('pending_username')
        otp_store.pop(mobile_e164, None)
        return jsonify({'success': True, 'message': 'OTP verified'})
    return jsonify({'error': 'Invalid OTP'}), 400

# Update set_password_mobile to store user in MongoDB
@app.route('/api/set-password-mobile', methods=['POST'])
def set_password_mobile():
    data = request.get_json()
    password = data.get('password')
    mobile = session.get('otp_verified_mobile')
    username = session.get('otp_verified_username')
    if not mobile or not username or not password:
        return jsonify({'error': 'OTP verification required and password required'}), 400
    users_collection.update_one(
        {'_id': username},
        {'$set': {'password': password, 'mobile': mobile, 'username': username}},
        upsert=True
    )
    session['user_id'] = username
    session['username'] = username
    session.pop('otp_verified_mobile', None)
    session.pop('otp_verified_username', None)
    return jsonify({'success': True, 'message': 'Password set and user logged in', 'user': {'id': username, 'username': username}})

# Update login to check MongoDB
@app.route('/api/login', methods=['POST'])
def login():
    data = request.get_json()
    username = data.get('username')
    password = data.get('password')
    if not username or not password:
        return jsonify({'error': 'Username/email and password required'}), 400
    user = users_collection.find_one({'_id': username})
    if not user:
        user = users_collection.find_one({'email': username})
    if not user:
        user = users_collection.find_one({'username': username})
    if user and user.get('password') == password:
        session['user_id'] = user.get('username', user.get('email', username))
        session['username'] = user.get('username', user.get('email', username))
        return jsonify({'success': True, 'message': 'Login successful', 'user': {'id': session['user_id'], 'username': session['username']}})
    return jsonify({'error': 'Invalid username/email or password'}), 401

# Update registered_users to list from MongoDB
@app.route('/api/registered-users', methods=['GET'])
def registered_users():
    users = list(users_collection.find({}, {'_id': 1}))
    return jsonify({'users': [u['_id'] for u in users]})

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy', 'models_loaded': mlp_model is not None})

@app.route('/predict', methods=['POST'])
@login_required
def predict():
    try:
        data = request.get_json()
        
        if not data or 'sequence' not in data:
            return jsonify({'error': 'No sequence provided'}), 400
        
        sequence = data['sequence'].strip()
        
        if not sequence:
            return jsonify({'error': 'Empty sequence provided'}), 400
        
        if len(sequence) < 10:
            return jsonify({'error': 'Sequence too short. Please provide at least 10 amino acids.'}), 400
        
        # Make prediction
        result = predict_sequence(sequence)
        
        return jsonify({
            'success': True,
            'result': result,
            'processed_sequence': preprocess_sequence(sequence)
        })
        
    except ValueError as ve:
        return jsonify({'error': str(ve)}), 400
    except Exception as e:
        logger.error(f"Error in predict endpoint: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/classes', methods=['GET'])
@login_required
def get_classes():
    """Get available prediction classes"""
    try:
        if label_encoder is None:
            return jsonify({'error': 'Models not loaded'}), 500
        
        return jsonify({
            'classes': label_encoder.classes_.tolist()
        })
    except Exception as e:
        logger.error(f"Error getting classes: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

# --- GOOGLE OAUTH ENDPOINTS ---
@app.route('/api/auth/google', methods=['GET'])
def google_auth():
    """Initiate Google OAuth flow - redirect to Google login page"""
    from urllib.parse import urlencode
    import requests
    
    # Get Google OAuth credentials from environment variables
    GOOGLE_CLIENT_ID = os.environ.get('GOOGLE_CLIENT_ID')
    
    if not GOOGLE_CLIENT_ID:
        return jsonify({'error': 'Google OAuth not configured. Please set GOOGLE_CLIENT_ID environment variable.'}), 500
    
    # Google OAuth authorization URL
    auth_url = 'https://accounts.google.com/o/oauth2/v2/auth'
    params = {
        'client_id': GOOGLE_CLIENT_ID,
        'redirect_uri': 'http://localhost:5000/api/auth/google/callback',
        'scope': 'openid email profile',
        'response_type': 'code',
        'access_type': 'offline',
        'prompt': 'select_account'  # This forces the account selection screen
    }
    
    auth_url_with_params = f"{auth_url}?{urlencode(params)}"
    return redirect(auth_url_with_params)

@app.route('/api/auth/google/callback', methods=['GET'])
def google_auth_callback():
    """Handle Google OAuth callback"""
    import requests
    from urllib.parse import urlencode
    
    try:
        code = request.args.get('code')
        error = request.args.get('error')
        
        if error:
            # If there's an error, redirect back to frontend
            return redirect(f"http://localhost:5173?auth=error&error={error}")
        
        if not code:
            return redirect(f"http://localhost:5173?auth=error&error=no_code")
        
        # Get Google OAuth credentials
        GOOGLE_CLIENT_ID = os.environ.get('GOOGLE_CLIENT_ID')
        GOOGLE_CLIENT_SECRET = os.environ.get('GOOGLE_CLIENT_SECRET')
        
        if not GOOGLE_CLIENT_ID or not GOOGLE_CLIENT_SECRET:
            return redirect(f"http://localhost:5173?auth=error&error=oauth_not_configured")
        
        # Exchange authorization code for access token
        token_url = 'https://oauth2.googleapis.com/token'
        token_data = {
            'client_id': GOOGLE_CLIENT_ID,
            'client_secret': GOOGLE_CLIENT_SECRET,
            'code': code,
            'grant_type': 'authorization_code',
            'redirect_uri': 'http://localhost:5000/api/auth/google/callback'
        }
        
        token_response = requests.post(token_url, data=token_data)
        token_response.raise_for_status()
        token_info = token_response.json()
        
        # Get user info using access token
        user_info_url = 'https://www.googleapis.com/oauth2/v2/userinfo'
        headers = {'Authorization': f"Bearer {token_info['access_token']}"}
        user_response = requests.get(user_info_url, headers=headers)
        user_response.raise_for_status()
        user_info = user_response.json()
        
        # Extract user information
        email = user_info.get('email')
        name = user_info.get('name', email)
        google_id = user_info.get('id')
        picture = user_info.get('picture')
        
        if not email:
            return redirect(f"http://localhost:5173?auth=error&error=no_email")
        
        # Store user in MongoDB
        users_collection.update_one(
            {'_id': email},
            {
                '$set': {
                    'email': email,
                    'name': name,
                    'google_id': google_id,
                    'picture': picture,
                    'last_login': datetime.utcnow()
                }
            },
            upsert=True
        )
        
        # Set session
        session['user_id'] = email
        session['username'] = name
        
        # Redirect to frontend with success and close window script
        return redirect(f"http://localhost:5173?auth=success&user={email}&close=true")
        
    except requests.RequestException as e:
        logger.error(f"Google OAuth error: {str(e)}")
        return redirect(f"http://localhost:5173?auth=error&error=oauth_failed")
    except Exception as e:
        logger.error(f"Google OAuth callback error: {str(e)}")
        return redirect(f"http://localhost:5173?auth=error&error=internal_error")

if __name__ == '__main__':
    # Load models on startup
    load_models()
    app.run(debug=True, host='0.0.0.0', port=5000)