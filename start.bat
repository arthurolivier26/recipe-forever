@echo off
chcp 65001 >nul
echo ================================
echo        MEAL PLANNER AI - LAUNCHER
echo ================================
echo.

:: Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python is not installed or not in the PATH
    echo.
    echo Please install Python from:
    echo https://www.python.org/downloads/
    echo.
    pause
    exit /b 1
)

echo [INFO] Python detected
python --version
echo.

:: Check if virtual environment exists
if not exist "venv\Scripts\activate.bat" (
    echo [1/5] Creating virtual environment...
    python -m venv venv
    if errorlevel 1 (
        echo [ERROR] Failed to create virtual environment
        pause
        exit /b 1
    )
    echo [OK] Virtual environment created
)

:: Activate virtual environment
echo [2/5] Activating virtual environment...
call venv\Scripts\activate.bat

:: Install / update dependencies
echo [3/5] Checking dependencies...
pip install -q --upgrade pip
pip install -q -r requirements.txt
if errorlevel 1 (
    echo [ERROR] Failed to install dependencies
    pause
    exit /b 1
)
echo [OK] Dependencies installed

:: Run setup script to verify data and model files
echo [4/5] Verifying data files...
python setup.py
echo.

:: Ready to launch
echo ================================
echo          READY TO START
echo ================================
echo.
echo The application will start:
echo   - Backend API : http://localhost:8000
echo   - Frontend UI : http://localhost:8501
echo.
pause

:: Launch FastAPI backend
echo [5/5] Starting servers...
start "FastAPI Backend" cmd /k "call venv\Scripts\activate.bat && python main.py"

:: Wait for backend to initialize
echo [INFO] Waiting for backend (5 seconds)...
timeout /t 5 /nobreak >nul

:: Launch Streamlit frontend
start "Streamlit Frontend" cmd /k "call venv\Scripts\activate.bat && streamlit run Home.py --server.port 8501"

echo.
echo ================================
echo      ✅ APPLICATION STARTED
echo ================================
echo.
echo Backend API : http://localhost:8000
echo Frontend UI : http://localhost:8501
echo API Docs    : http://localhost:8000/docs
echo.
echo Press any key to stop both servers...
pause >nul

:: Kill server processes
taskkill /FI "WINDOWTITLE eq FastAPI Backend*" /T /F >nul 2>&1
taskkill /FI "WINDOWTITLE eq Streamlit Frontend*" /T /F >nul 2>&1

echo.
echo Servers stopped.
pause
