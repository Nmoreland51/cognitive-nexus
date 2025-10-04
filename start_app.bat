@echo off
echo 🧠 Cognitive Nexus AI - Starting...
echo ==================================

echo.
echo 🚀 Launching your AI assistant...
echo 📱 The app will open in your browser
echo ⏹️  Press Ctrl+C to stop the server
echo.

REM Create a temporary file with empty email
echo. > temp_email.txt

REM Launch Streamlit with the email file as input
python -m streamlit run cognitive_nexus_ai.py --server.port 8501 --server.address localhost < temp_email.txt

REM Clean up
del temp_email.txt

pause
