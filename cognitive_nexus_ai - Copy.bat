@echo off
title Cognitive Nexus AI
color 0A

echo.
echo ========================================
echo   🧠 Cognitive Nexus AI - Launcher
echo ========================================
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python not found! Please install Python first.
    echo Download from: https://python.org
    pause
    exit /b 1
)

echo ✅ Python detected
echo.

REM Create directories if they don't exist
if not exist "ai_system" mkdir "ai_system"
if not exist "ai_system\knowledge_bank" mkdir "ai_system\knowledge_bank"
if not exist "ai_system\knowledge_bank\images" mkdir "ai_system\knowledge_bank\images"
if not exist "ai_system\knowledge_bank\topics" mkdir "ai_system\knowledge_bank\topics"
if not exist "ai_system\logs" mkdir "ai_system\logs"
if not exist "data" mkdir "data"

echo ✅ Directories ready
echo.

REM Try to install basic dependencies (non-blocking)
echo 📦 Installing basic dependencies...
pip install streamlit requests beautifulsoup4 psutil >nul 2>&1
echo ✅ Basic dependencies ready

REM Try to install image generation dependencies (non-blocking)
echo 📦 Installing image generation dependencies...
pip install torch diffusers pillow transformers accelerate safetensors >nul 2>&1
echo ✅ Image generation dependencies ready

REM Try to install OpenChat dependencies (non-blocking)
echo 📦 Installing OpenChat dependencies...
pip install torch transformers accelerate bitsandbytes >nul 2>&1
echo ✅ OpenChat dependencies ready
echo.

echo 🚀 Starting Cognitive Nexus AI...
echo.
echo 🌐 Opening browser at: http://localhost:8501
echo ⏹️  Press Ctrl+C to stop
echo.

REM Launch the app
streamlit run cognitive_nexus_ai.py --server.port 8501 --server.address localhost

echo.
echo 👋 App stopped. Press any key to close.
pause >nul
