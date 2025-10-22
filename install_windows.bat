color a

@echo off
echo ============================================
echo Mini-LLM v2 - Instalacion para Windows
echo ============================================
echo.

REM Verifica Python
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python no encontrado
    echo Descarga Python 3.10+ de python.org
    pause
    exit /b 1
)

echo Verificando version de Python...
python -c "import sys; exit(0 if sys.version_info >= (3,10) else 1)"
if errorlevel 1 (
    echo ERROR: Se requiere Python 3.10 o superior
    pause
    exit /b 1
)

echo.
echo Instalando PyTorch con CUDA 11.8...
pip install torch --index-url https://download.pytorch.org/whl/cu118

echo.
echo Instalando dependencias restantes...
pip install -r requirements.txt

echo.
echo Verificando instalacion...
python check_dependencies.py

if errorlevel 1 (
    echo.
    echo ERROR: Problemas con las dependencias
    pause
    exit /b 1
)

echo.
echo ============================================
echo Instalacion completada exitosamente!
echo ============================================
echo.
echo Proximos pasos:
echo   1. Ejecuta: python interactive_cli.py
echo   2. O: python check_dependencies.py
echo.
pause