@echo off
rem Abre el organizador de imágenes. Cierra esta ventana para detenerlo.
chcp 65001 >nul
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo Falta el entorno de Python. Sigue los pasos de instalacion del README.
    pause
    exit /b 1
)
".venv\Scripts\python.exe" run.py
pause
