# 🧬 T2D Insulin Prediction Tool - Setup Guide

## 📋 Overview

This project uses your custom trained model (from `ourprogen-mlp.ipynb`) and provides two REST backends:
- **Combined Backend** (`backend/combined_server.py`, default, port 5001 & 5000)
- **Legacy/Archived servers**: see `backend/app_legacy.py`, `backend/model_server_archived.py` if needed
- **Frontend** (`src/`, React, port 5173)

## 🏗️ Project Structure

```
PS-3-1/
├── backend/
│   ├── combined_server.py      # Combined authentication + prediction server (main)
│   ├── app_legacy.py           # Legacy/deprecated combined server (ignore)
│   ├── model_server_archived.py # Archived model-only server
│   ├── requirements.txt
│   ├── setup.py                # Backend setup (creates folders, prints setup info)
│   └── models/
│        ├── best_mlp_medium_adv.pth   # MLP classifier
│        ├── label_encoder.pkl         # Label encoder
│        ├── pca_model.pkl             # PCA model
│        ├── Protein_to_Smile.pt       # Protein to SMILES (optional)
│        ├── Sequence_Generator.pt     # Sequence Generator (optional)
│        └── custom_protein_lm.pt      # (You must add this, your trained ProteinLM)
├── src/                         # React frontend
├── start_servers.bat            # Start all servers (Windows, recommended)
├── setup_custom_model.py        # Check/copy model file helper
├── setup_credentials.py         # Generate backend/.env template
├── setup_mongodb.py             # Test/setup MongoDB
```

## 🚀 Quick Start

### Windows
```bash
# 1. Start combined (auth+model) server (separate CMD windows)
start_servers.bat

# 2. In another terminal, start frontend
npm run dev
```

### Manual (Advanced)
```bash
# 1. Auth+model server (default, combined):
cd backend
python combined_server.py

# To use only old legacy servers (not recommended):
# python app_legacy.py   # Authentication + model (legacy interface)
# python model_server_archived.py # Model only (legacy interface)

# 2. Start frontend
npm run dev
```

## 🔧 Setup Your Model Files

1. Copy your custom ProteinLM checkpoint:
   ```bash
   # Place your trained model here after running your ownprogen/mlp training
   cp /path/to/your/custom_protein_lm.pt backend/models/
   ```
2. Verify all required files are present with:
   ```bash
   python setup_custom_model.py
   ```
   Required model files (in `backend/models/`):
   - `best_mlp_medium_adv.pth` (MLP classifier)
   - `label_encoder.pkl`
   - `pca_model.pkl`
   - `custom_protein_lm.pt` (**you provide this**)
   - Optional: `Protein_to_Smile.pt`, `Sequence_Generator.pt`

## 🌐 API Endpoints

- **Combined server:** http://localhost:5001 (Auth), http://localhost:5000 (Model)
    - `/api/register`, `/api/login` ... (see code for full)
    - `/predict`, `/classes`, `/health`
    - `/api/auth/google`  (Google OAuth setup, see GOOGLE_OAUTH_SETUP.md)
- **Frontend:** http://localhost:5173

## ⚙️ Environment Configuration (.env)

1. Create a `.env` in `backend/` (script available: `python setup_credentials.py`). Fill in Google, MongoDB, Email, Twilio (see scripts and GOOGLE_OAUTH_SETUP.md for guidance):

```
GOOGLE_CLIENT_ID=...
GOOGLE_CLIENT_SECRET=...
SECRET_KEY=...
MONGO_URI=mongodb://localhost:27017/
EMAIL_ADDRESS=...
EMAIL_PASSWORD=...
TWILIO_ACCOUNT_SID=...
TWILIO_AUTH_TOKEN=...
TWILIO_PHONE_NUMBER=...
DEFAULT_COUNTRY_CODE=+91
DEV_FAKE_OTP=false
```

## 🛠️ Troubleshooting

- **Model loading fails?** Use `python setup_custom_model.py` to check missing files.
- **MongoDB required?** Use `python setup_mongodb.py` to test/check connection.
- **.env issues?** Re-run `python setup_credentials.py`.
- **OAuth issues?** See `GOOGLE_OAUTH_SETUP.md` and restart backend after changing credentials.

## 🔥 Useful Scripts

- `start_servers.bat`: Start all backends (cmd windows) [Recommended, Windows]
- `python setup_custom_model.py`: Check you have all model files after export
- `python setup_credentials.py`: Template for `.env`
- `python setup_mongodb.py`: Test MongoDB

## 🎉 Next Steps

1. Copy `custom_protein_lm.pt` to `backend/models/`
2. Run `python setup_custom_model.py` to verify setup
3. Start servers with `start_servers.bat`
4. Start React frontend with `npm run dev`
5. Open `http://localhost:5173` and register/login
6. Enjoy reliable T2D Insulin Prediction!

