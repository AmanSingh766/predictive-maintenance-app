@echo off
echo ================================================
echo  INDUSTRIAL AI PREDICTIVE MAINTENANCE SYSTEM
echo ================================================

echo.
echo Activating virtual environment...
call backend\venv\Scripts\activate

echo.
echo Starting backend server...
start cmd /k "cd backend && python app.py"

echo.
echo Waiting for server to start...
timeout /t 3 > nul

echo.
echo Opening frontend dashboard...
start frontend\index.html

echo.
echo ================================================
echo  SYSTEM STARTED SUCCESSFULLY
echo ================================================
pause
