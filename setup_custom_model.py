#!/usr/bin/env python3
"""
Setup script to copy custom model files to backend/models directory
"""

import os
import shutil
from pathlib import Path

def setup_custom_model():
    """Copy custom model files to backend/models directory"""
    
    print("🔧 Setting up custom model files...")
    
    # Create models directory if it doesn't exist
    models_dir = Path("backend/models")
    models_dir.mkdir(exist_ok=True)
    
    # Files to copy (you'll need to provide the actual paths)
    files_to_copy = {
        # Update these paths to match your actual file locations
        "custom_protein_lm.pt": "path/to/your/custom_protein_lm.pt",  # Your custom ProteinLM checkpoint
        "best_mlp_medium_adv.pth": "backend/models/best_mlp_medium_adv.pth",  # Already exists
        "label_encoder.pkl": "backend/models/label_encoder.pkl",  # Already exists
        "pca_model.pkl": "backend/models/pca_model.pkl",  # Already exists
    }
    
    print("📁 Current model files in backend/models/:")
    for file in models_dir.iterdir():
        if file.is_file():
            print(f"  ✅ {file.name} ({file.stat().st_size} bytes)")
    
    print("\n📋 Required files for custom model:")
    print("  - custom_protein_lm.pt (Custom ProteinLM checkpoint)")
    print("  - best_mlp_medium_adv.pth (MLP classifier)")
    print("  - label_encoder.pkl (Label encoder)")
    print("  - pca_model.pkl (PCA model)")
    
    print("\n⚠️  Note: You need to provide the custom_protein_lm.pt file")
    print("   This should be the checkpoint from your ourprogen-mlp.ipynb training")
    print("   The file should contain the 'model_state' key with your custom ProteinLM weights")
    
    # Check if all required files exist
    missing_files = []
    for file_name in ["best_mlp_medium_adv.pth", "label_encoder.pkl", "pca_model.pkl"]:
        file_path = models_dir / file_name
        if not file_path.exists():
            missing_files.append(file_name)
    
    if missing_files:
        print(f"\n❌ Missing files: {', '.join(missing_files)}")
        print("   Please ensure these files are in backend/models/ directory")
    else:
        print("\n✅ All required model files are present!")
    
    print("\n🎯 Next steps:")
    print("1. Copy your custom_protein_lm.pt to backend/models/")
    print("2. Update the CKPT_PATH in model_server.py to point to the correct file")
    print("3. Run the servers using start_servers.bat or start_servers.py")

if __name__ == "__main__":
    setup_custom_model()

