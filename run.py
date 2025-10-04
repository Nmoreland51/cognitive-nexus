#!/usr/bin/env python3
"""
Cognitive Nexus AI - Executable Wrapper
This script is used by PyInstaller to create a standalone executable.
It launches the Streamlit app exactly as if you ran 'streamlit run cognitive_nexus_ai.py'
"""

import sys
import os
import subprocess
import webbrowser
import time
from pathlib import Path

def main():
    """Main entry point for the executable."""
    try:
        # Get the directory where the executable is located
        if getattr(sys, 'frozen', False):
            # Running as compiled executable
            base_path = Path(sys._MEIPASS)
            app_path = base_path / "cognitive_nexus_ai.py"
        else:
            # Running as script
            base_path = Path(__file__).parent
            app_path = base_path / "cognitive_nexus_ai.py"
        
        # Verify the main app file exists
        if not app_path.exists():
            print(f"ERROR: cognitive_nexus_ai.py not found at {app_path}")
            input("Press Enter to exit...")
            sys.exit(1)
        
        # Set the working directory to the app directory
        os.chdir(base_path)
        
        print("🚀 Starting Cognitive Nexus AI...")
        print("📱 The app will open in your default browser")
        print("🌐 URL: http://localhost:8501")
        print("⏹️  Press Ctrl+C to stop the application")
        print("-" * 50)
        
        # Start the Streamlit app
        process = subprocess.Popen([
            sys.executable, "-m", "streamlit", "run", 
            str(app_path),
            "--server.port", "8501",
            "--server.address", "localhost",
            "--server.headless", "true",
            "--browser.gatherUsageStats", "false"
        ])
        
        # Wait a moment for Streamlit to start
        time.sleep(3)
        
        # Open browser automatically
        try:
            webbrowser.open("http://localhost:8501")
        except Exception:
            pass  # Browser opening is optional
        
        # Wait for the process to complete
        process.wait()
        
    except KeyboardInterrupt:
        print("\n🛑 Shutting down Cognitive Nexus AI...")
        if 'process' in locals():
            process.terminate()
        sys.exit(0)
    except Exception as e:
        print(f"❌ Error starting Cognitive Nexus AI: {e}")
        print(f"📁 Current directory: {os.getcwd()}")
        print(f"📄 Looking for: {app_path}")
        input("Press Enter to exit...")
        sys.exit(1)

if __name__ == "__main__":
    main()