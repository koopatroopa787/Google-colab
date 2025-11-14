@echo off
REM Automated Box2D installer for Windows
REM This script will download and install the pre-built Box2D wheel

echo ========================================
echo Box2D Installer for Windows
echo ========================================
echo.

REM Check Python version
python --version
if %errorlevel% neq 0 (
    echo Error: Python is not installed or not in PATH
    pause
    exit /b 1
)

echo.
echo Detecting Python version...
for /f "tokens=2" %%i in ('python --version 2^>^&1') do set PYVER=%%i
echo Python version: %PYVER%

REM Extract major.minor version (e.g., 3.11)
for /f "tokens=1,2 delims=." %%a in ("%PYVER%") do (
    set PYMAJOR=%%a
    set PYMINOR=%%b
)

echo Major: %PYMAJOR%, Minor: %PYMINOR%
echo.

REM Determine which wheel to download
if "%PYMAJOR%"=="3" (
    if "%PYMINOR%"=="11" (
        set WHEEL_URL=https://download.lfd.uci.edu/pythonlibs/archived/Box2D-2.3.10-cp311-cp311-win_amd64.whl
        set WHEEL_NAME=Box2D-2.3.10-cp311-cp311-win_amd64.whl
    ) else if "%PYMINOR%"=="10" (
        set WHEEL_URL=https://download.lfd.uci.edu/pythonlibs/archived/Box2D-2.3.10-cp310-cp310-win_amd64.whl
        set WHEEL_NAME=Box2D-2.3.10-cp310-cp310-win_amd64.whl
    ) else if "%PYMINOR%"=="9" (
        set WHEEL_URL=https://download.lfd.uci.edu/pythonlibs/archived/Box2D-2.3.10-cp39-cp39-win_amd64.whl
        set WHEEL_NAME=Box2D-2.3.10-cp39-cp39-win_amd64.whl
    ) else (
        echo Error: Unsupported Python version. Need Python 3.9, 3.10, or 3.11
        pause
        exit /b 1
    )
) else (
    echo Error: Need Python 3.x
    pause
    exit /b 1
)

echo.
echo Downloading Box2D wheel for Python %PYMAJOR%.%PYMINOR%...
echo URL: %WHEEL_URL%
echo.

REM Download using PowerShell
powershell -Command "& {Invoke-WebRequest -Uri '%WHEEL_URL%' -OutFile '%WHEEL_NAME%'}"

if %errorlevel% neq 0 (
    echo.
    echo Error: Failed to download wheel file
    echo.
    echo Please download manually from:
    echo %WHEEL_URL%
    echo.
    echo Then run: pip install %WHEEL_NAME%
    pause
    exit /b 1
)

echo.
echo ✓ Download successful!
echo.
echo Installing Box2D wheel...
python -m pip install %WHEEL_NAME%

if %errorlevel% neq 0 (
    echo.
    echo Error: Failed to install Box2D
    pause
    exit /b 1
)

echo.
echo ✓ Box2D installed successfully!
echo.
echo Installing gymnasium[box2d]...
python -m pip install "gymnasium[box2d]"

echo.
echo ========================================
echo Installation Complete!
echo ========================================
echo.
echo Testing installation...
python -c "import Box2D; print('✓ Box2D works!')"

if %errorlevel% equ 0 (
    echo.
    echo ✓ All set! You can now use BipedalWalker!
    echo.
    echo Run: python rl_bipedal_walker\app.py
) else (
    echo.
    echo ✗ Import test failed. Please check the installation.
)

echo.
echo Cleaning up...
del %WHEEL_NAME%

echo.
pause
