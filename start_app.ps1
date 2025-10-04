# Cognitive Nexus AI Launcher
Write-Host "🧠 Cognitive Nexus AI - Starting..." -ForegroundColor Cyan
Write-Host "==================================" -ForegroundColor Cyan

Write-Host ""
Write-Host "🚀 Launching your AI assistant..." -ForegroundColor Green
Write-Host "📱 The app will open in your browser at http://localhost:8501" -ForegroundColor Yellow
Write-Host "⏹️  Press Ctrl+C to stop the server" -ForegroundColor Red
Write-Host ""

# Start the Streamlit app
$process = Start-Process -FilePath "python" -ArgumentList "-m", "streamlit", "run", "cognitive_nexus_ai.py", "--server.port", "8501", "--server.address", "localhost" -PassThru -NoNewWindow

# Wait a moment for the app to start
Start-Sleep -Seconds 10

# Open the browser
Write-Host "🌐 Opening browser..." -ForegroundColor Green
Start-Process "http://localhost:8501"

# Wait for the process
$process.WaitForExit()
