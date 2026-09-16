@echo off
setlocal

cd /d "%~dp0"
set "ROOT=%~dp0"
set "PYTHON=%ROOT%.venv\Scripts\python.exe"

if not exist "%PYTHON%" (
    echo [ERROR] Python virtual environment not found at:
    echo         %ROOT%.venv
    echo.
    echo Create it with: python -m venv .venv
    exit /b 1
)

if not exist "%ROOT%frontend\node_modules" (
    echo [ERROR] Frontend dependencies not found.
    echo Run: cd frontend ^&^& npm install
    exit /b 1
)

echo Starting DefectFill backend at http://localhost:8000 ...
start "DefectFill Backend" /D "%ROOT%" cmd /k ""%PYTHON%" -m uvicorn defectfill.backend.app:app --host 0.0.0.0 --port 8000"

echo Starting DefectFill frontend at http://localhost:3000 ...
start "DefectFill Frontend" /D "%ROOT%frontend" cmd /k "npm run dev"

echo.
echo DefectFill is starting. Open http://localhost:3000 in your browser.
endlocal
