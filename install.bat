@echo off
REM 1) Create venv if it doesn’t exist
IF NOT EXIST ".venv\Scripts\python.exe" (
  python -m venv .venv
)

REM 2) Activate the venv
call .venv\Scripts\activate.bat

REM 3) Upgrade pip & install requirements
python -m pip install --upgrade pip
pip install -r requirements.txt

echo.
echo ✅ Setup complete!
echo To run:
echo    .venv\Scripts\activate.bat
echo    python SHG_Analysis.py
pause