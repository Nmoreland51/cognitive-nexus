@echo off
echo 🧠 Cognitive Nexus AI - Launcher
echo ================================

echo.
echo 🔄 Refreshing environment...
set PATH=%PATH%;C:\Users\%USERNAME%\AppData\Local\Programs\Python\Python312;C:\Users\%USERNAME%\AppData\Local\Programs\Python\Python312\Scripts

echo.
echo 🚀 Starting your AI assistant...
echo 📱 The app will open in your browser at http://localhost:8501
echo ⏹️  Press Ctrl+C to stop the server
echo.

python -m streamlit run cognitive_nexus_ai.py --server.port 8501 --server.address localhost --server.headless true

pause
