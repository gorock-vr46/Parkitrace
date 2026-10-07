@echo off
cd /d "%~dp0"
if not exist venv\Scripts\python.exe (
  echo Virtual environment not found.
  echo Create it first with: python -m venv venv
  pause
  exit /b 1
)
call venv\Scripts\activate.bat
python app.py
pause
