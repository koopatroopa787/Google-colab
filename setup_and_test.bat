@echo off
REM Windows batch script to run setup and testing
REM This is a wrapper for the Python setup script

echo ========================================
echo AI/ML Projects - Setup and Testing
echo ========================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Error: Python is not installed or not in PATH
    echo Please install Python 3.8+ from https://www.python.org/
    pause
    exit /b 1
)

REM Run the setup script
python setup_and_test.py

REM Pause to see results
pause
