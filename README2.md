# X-Ray Diagnosis Project - Windows Installation Guide

## Prerequisites
- Python 3.8+ (64-bit)
- Windows 10/11
- Stable Internet Connection

## Installation Methods

### Method 1: Direct Installation
```cmd
# Open Command Prompt
git clone <repository_url>
cd X-Ray-Diagnosis-main

# Install dependencies
pip install -r requirements.txt

# Run application
streamlit run app.py

Method 2: Virtual Environment (Recommended)
# Open Command Prompt

# Clone Repository
git clone <repository_url>
cd X-Ray-Diagnosis-main

# Create Virtual Environment
python -m venv xray_env

# Activate Virtual Environment
xray_env\Scripts\activate
# py -3.11 -m venv xray_env

# Install Dependencies
pip install -r requirements.txt

# Run Application
streamlit run app.py

Comprehensive Troubleshooting Guide
Python Installation Issues
1. Python Not Found
   : Ensure Python is installed
   : Add Python to Windows PATH
   : Verify installation: python --version
2. Incorrect Python Version
   : Required: Python 3.8+
   : Check version: python --version
   : Download from Python Official Site

Dependency Installation Problems
1. pip Installation Errors
# Upgrade pip
python -m pip install --upgrade pip

# Force reinstall dependencies
pip install -r requirements.txt --upgrade --force-reinstall

2. Missing Dependencies
# Manual package installation
pip install streamlit torch torchvision ultralytics pillow


Streamlit Specific Errors

1. Streamlit Import Errors
# Reinstall Streamlit
pip uninstall streamlit
pip install streamlit

2. Port Conflicts
streamlit run app.py --server.port 8501