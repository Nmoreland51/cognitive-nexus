#!/usr/bin/env python3
"""
Cognitive Nexus AI Launcher
This script will launch the Streamlit app with proper configuration
"""

import subprocess
import sys
import time
import webbrowser
from pathlib import Path

def main():
    print("🧠 Cognitive Nexus AI - Launcher")
    print("=" * 40)
    
    # Check if the main file exists
    main_file = Path("cognitive_nexus_ai.py")
    if not main_file.exists():
        print("❌ cognitive_nexus_ai.py not found!")
        return 1
    
    print("✅ Found cognitive_nexus_ai.py")
    
    # Launch Streamlit
    print("🚀 Launching Streamlit app...")
    print("📱 The app will open in your browser at http://localhost:8501")
    print("⏹️  Press Ctrl+C to stop the server")
    print("-" * 40)
    
    try:
        # Use subprocess to run streamlit
        process = subprocess.Popen([
            sys.executable, "-m", "streamlit", "run", 
            str(main_file),
            "--server.port", "8501",
            "--server.address", "localhost",
            "--server.headless", "true"
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        
        # Wait a moment for the server to start
        time.sleep(5)
        
        # Open browser
        print("🌐 Opening browser...")
        webbrowser.open("http://localhost:8501")
        
        # Wait for the process
        process.wait()
        
    except KeyboardInterrupt:
        print("\n\n👋 Stopping Cognitive Nexus AI...")
        process.terminate()
        return 0
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
