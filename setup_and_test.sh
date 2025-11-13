#!/bin/bash
# Unix/Linux/macOS shell script to run setup and testing
# This is a wrapper for the Python setup script

echo "========================================"
echo "AI/ML Projects - Setup and Testing"
echo "========================================"
echo ""

# Check if Python 3 is installed
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed"
    echo "Please install Python 3.8+ from your package manager"
    exit 1
fi

# Run the setup script
python3 setup_and_test.py

# Exit with the same code as the Python script
exit $?
