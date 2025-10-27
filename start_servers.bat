@echo off
echo 🚀 Starting T2D Insulin Prediction Tool
echo ==================================================

echo 🔐 Starting Authentication Server on port 5001...
start "Auth Server" cmd /k "cd backend && python auth_server.py"

timeout /t 3 /nobreak >nul

echo 🧬 Starting Model Prediction Server on port 5000...
start "Model Server" cmd /k "cd backend && python model_server.py"

timeout /t 3 /nobreak >nul

echo.
echo ✅ Both servers are starting up...
echo 🔐 Authentication Server: http://localhost:5001
echo 🧬 Model Prediction Server: http://localhost:5000
echo 🌐 Frontend: http://localhost:5173
echo.
echo Press any key to close this window...
pause >nul

