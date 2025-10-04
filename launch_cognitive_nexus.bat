@echo off
echo ========================================
echo   Cognitive Nexus AI - Launcher
echo ========================================
echo.

:: Check if executable exists
if not exist "dist\CognitiveNexusAI\CognitiveNexusAI.exe" (
    echo ERROR: Executable not found!
    echo Please run build_executable.bat first to create the executable.
    echo.
    pause
    exit /b 1
)

:: Launch the executable
echo Starting Cognitive Nexus AI...
echo The app will open in your browser at http://localhost:8501
echo.
echo Press Ctrl+C to stop the application
echo.

cd /d "dist\CognitiveNexusAI"
CognitiveNexusAI.exe

pause