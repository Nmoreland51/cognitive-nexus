@echo off
echo 🧠 Cognitive Nexus AI - Launcher
echo ================================

echo.
echo 🚀 Starting the application...
echo 📱 The app will open in your browser at http://localhost:8501
echo ⏹️  Press Ctrl+C to stop the server
echo.

REM Launch Streamlit with automatic email skip
echo. | python -m streamlit run cognitive_nexus_ai.py --server.port 8501 --server.address localhost

pause
