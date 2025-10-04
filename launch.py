#!/usr/bin/env python3
"""
Cognitive Nexus AI Launcher
"""

import subprocess
import sys
import time
import webbrowser
import os

def main():
    print("🧠 Cognitive Nexus AI - Launcher")
    print("=" * 40)
    
    # Set environment variable to skip email prompt
    os.environ['STREAMLIT_BROWSER_GATHER_USAGE_STATS'] = 'false'
    
    print("🚀 Starting Streamlit app...")
    print("📱 The app will open in your browser at http://localhost:8501")
    print("⏹️  Press Ctrl+C to stop the server")
    print("-" * 40)
    
    try:
        # Launch Streamlit
        process = subprocess.Popen([
            sys.executable, "-m", "streamlit", "run", 
            "cognitive_nexus_ai.py",
            "--server.port", "8501",
            "--server.address", "localhost",
            "--server.headless", "true",
            "--browser.gatherUsageStats", "false"
        ])
        
        # Wait for the server to start
        print("⏳ Waiting for server to start...")
        time.sleep(10)
        
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
