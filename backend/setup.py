#!/usr/bin/env python3
"""
Setup script for the protein sequence classifier backend.
This script creates the necessary directory structure and provides
instructions for setting up the virtual environment.
"""

import os
import sys

def create_directories():
    """Create necessary directories for the Flask backend"""
    directories = [
        'models',
        'logs'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✅ Created directory: {directory}")

def print_instructions():
    """Print setup instructions"""
    instructions = """
🔧 SETUP INSTRUCTIONS:

1. Create and activate virtual environment:
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\\Scripts\\activate

2. Install dependencies:
   pip install -r requirements.txt

3. Copy your trained model files to the models/ directory:
   - Copy best_mlp_medium_adv.pth to models/best_mlp_medium_adv.pth
   - Copy label_encoder.pkl to models/label_encoder.pkl  
   - Copy pca_model.pkl to models/pca_model.pkl

4. Run the Flask server:
   python app.py

5. The API will be available at: http://localhost:5000

📁 Required model files in models/ directory:
   ✓ best_mlp_medium_adv.pth (PyTorch MLP model weights)
   ✓ label_encoder.pkl (Scikit-learn label encoder)
   ✓ pca_model.pkl (PCA dimensionality reduction model)

🌐 API Endpoints:
   GET  /health    - Check server health
   POST /predict   - Make predictions (send JSON: {"sequence": "MVLSPADKTNVK..."})
   GET  /classes   - Get available prediction classes

⚠️  Note: First startup will download the ProGen2 model (~2GB) from Hugging Face.
    Make sure you have sufficient disk space and internet connection.
    """
    print(instructions)

if __name__ == "__main__":
    print("🧬 Protein Sequence Classifier - Backend Setup")
    print("=" * 50)
    
    create_directories()
    print_instructions()
    
    print("\n🎉 Setup complete! Follow the instructions above to start the server.")