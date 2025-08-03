#!/usr/bin/env python3
"""
Simple script to run the Fake News Detector Streamlit app
"""

import subprocess
import sys
import os

def main():
    """Run the Streamlit app"""
    print("🔍 Starting Fake News Detector...")
    print("📁 Working directory:", os.getcwd())
    
    try:
        # Run streamlit
        cmd = [sys.executable, "-m", "streamlit", "run", "streamlit_app.py", 
               "--server.headless", "true", "--server.port", "8501"]
        
        print("🚀 Running command:", " ".join(cmd))
        print("🌐 App will be available at: http://localhost:8501")
        print("⏹️  Press Ctrl+C to stop the server")
        print("-" * 50)
        
        subprocess.run(cmd)
        
    except KeyboardInterrupt:
        print("\n👋 Shutting down the server...")
    except Exception as e:
        print(f"❌ Error starting the app: {e}")
        print("💡 Make sure you have installed the requirements:")
        print("   pip install -r requirements.txt")

if __name__ == "__main__":
    main()