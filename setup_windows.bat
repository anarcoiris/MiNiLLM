@echo off
REM ============================================
REM Mini-LLM v2 - Setup y Test para Windows
REM ============================================

echo.
echo ============================================
echo       Mini-LLM v2 - Setup Windows
echo ============================================
echo.

REM Verifica que estemos en el directorio correcto
if not exist "main.py" (
    echo ERROR: main.py no encontrado
    echo Ejecuta este script desde el directorio del proyecto
    pause
    exit /b 1
)

REM Verifica Python
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python no encontrado en el PATH
    echo.
    echo Descarga Python 3.10+ de python.org
    echo Asegurate de marcar "Add Python to PATH" durante la instalacion
    pause
    exit /b 1
)

echo [1/4] Verificando Python...
python --version

echo.
echo [2/4] Instalando/Actualizando dependencias...
echo.

REM Actualiza pip
python -m pip install --upgrade pip

REM Instala PyTorch para Windows con CUDA
echo Instalando PyTorch con CUDA 11.8...
python -m pip install torch --index-url https://download.pytorch.org/whl/cu118

REM Instala resto de dependencias
echo Instalando dependencias restantes...
python -m pip install -r requirements.txt

echo.
echo [3/4] Aplicando fix para Windows...
echo.
python fix_windows_cli.py

if errorlevel 1 (
    echo.
    echo ADVERTENCIA: El fix de Windows no se aplico correctamente
    echo El CLI interactivo puede tener problemas con caracteres especiales
    echo Pero puedes continuar usando main.py directamente
    echo.
    pause
)

echo.
echo [4/4] Ejecutando smoke test completo...
echo.
python smoke_test.py --verbose

if errorlevel 1 (
    echo.
    echo ============================================
    echo     ADVERTENCIA: Algunos tests fallaron
    echo ============================================
    echo.
    echo Revisa los errores arriba.
    echo Puedes intentar arreglarlos manualmente o continuar si no son criticos.
    echo.
) else (
    echo.
    echo ============================================
    echo      Todo listo! Sistema funcionando
    echo ============================================
    echo.
)

echo.
echo PROXIMOS PASOS:
echo   1. Para CLI interactivo: python interactive_cli.py
echo   2. Para uso directo: python main.py --help
echo   3. Re-ejecutar test: python smoke_test.py
echo.

pause
