#!/usr/bin/env python3
"""
Startup script to run both authentication and model prediction servers
"""

import subprocess
import sys
import time
from pathlib import Path

def run_combined_server():
    """Run the combined auth + model server"""
    print("🔐🧬 Starting Combined Server on port 5001...")
    backend_dir = Path(__file__).parent / "backend"
    subprocess.run([sys.executable, "combined_server.py"], cwd=str(backend_dir))

def main():
    print("🚀 Starting T2D Insulin Prediction Tool")
    print("=" * 50)
    
    backend = Path("backend")
    if not backend.exists():
        print("❌ Backend directory not found!")
        sys.exit(1)
    if not (backend / "combined_server.py").exists():
        print("❌ combined_server.py not found in backend/")
        sys.exit(1)
    
    try:
        run_combined_server()
    except KeyboardInterrupt:
        print("\n\n🛑 Shutting down server...")
        sys.exit(0)

if __name__ == "__main__":
    main()
