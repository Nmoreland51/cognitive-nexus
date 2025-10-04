@echo off
echo ========================================
echo   Cognitive Nexus AI - Simple Demo
echo ========================================
echo.

:: Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python from https://python.org
    pause
    exit /b 1
)

:: Check if demo file exists
if not exist "cognitive_nexus_simple_demo.py" (
    echo ERROR: cognitive_nexus_simple_demo.py not found
    echo Please run this script from the project directory
    pause
    exit /b 1
)

:: Install streamlit if not present
echo Checking Streamlit installation...
python -c "import streamlit" >nul 2>&1
if %errorlevel% neq 0 (
    echo Installing Streamlit...
    python -m pip install streamlit --quiet
)

echo Starting Cognitive Nexus Simple Demo...
echo The app will open in your browser at http://localhost:8502
echo.
echo Press Ctrl+C to stop the application
echo.

:: Launch the simple demo on port 8502 to avoid conflicts
python -m streamlit run cognitive_nexus_simple_demo.py --server.port 8502 --server.address localhost

pause
