# Project Dependencies Guide

This document outlines the dependencies required for running this project across different operating systems.

## Common Dependencies

These dependencies are required for all operating systems: 
python
pandas==2.1.4
numpy>=1.21.6,<1.25.0
scikit-learn>=1.0.2
google-cloud-artifact-registry>=3.0.0
fairlearn>=0.7.0
xgboost>=1.7.3
docker>=6.0.0
matplotlib>=3.5.0
seaborn>=0.12.0
joblib>=1.1.0

## Operating System Specific Dependencies

### macOS

1. **OpenMP Library**
   - Required for XGBoost
   - Install using Homebrew:
   ```bash
   brew install libomp
   ```

2. **Installation Steps**
   ```bash
   # Install Homebrew if not already installed
   /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
   
   # Install dependencies
   brew install libomp
   
   # Create and activate conda environment
   conda create -n myenv python=3.9
   conda activate myenv
   
   # Install requirements
   pip install -r requirements_mac.txt
   ```

### Windows

1. **Microsoft Visual C++ Redistributable**
   - Required for XGBoost
   - Download from [Microsoft's official website](https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist)
   - Install both x64 and x86 versions if unsure

2. **Installation Steps**
   ```bash
   # Create and activate conda environment
   conda create -n myenv python=3.9
   conda activate myenv
   
   # Install XGBoost through conda first
   conda install -c conda-forge xgboost
   
   # Install other requirements
   pip install -r requirements_windows.txt
   ```

### Linux

1. **OpenMP Library**
   - Required for XGBoost
   - Installation commands:
     ```bash
     # For Ubuntu/Debian
     sudo apt-get update
     sudo apt-get install libgomp1
     
     # For CentOS/RHEL
     sudo yum install libgomp
     ```

2. **Installation Steps**
   ```bash
   # Create and activate conda environment
   conda create -n myenv python=3.9
   conda activate myenv
   
   # Install requirements
   pip install -r requirements_linux.txt
   ```

## Troubleshooting Common Issues

### XGBoost Issues

1. **macOS**
   - If you encounter `Library not loaded: @rpath/libomp.dylib` error:
     ```bash
     brew install libomp
     ```

2. **Windows**
   - If you get DLL errors:
     - Ensure Visual C++ Redistributable is installed
     - Try reinstalling XGBoost: `conda install -c conda-forge xgboost`

3. **Linux**
   - If you get `libgomp.so.1: cannot open shared object file`:
     ```bash
     sudo apt-get install libgomp1  # Ubuntu/Debian
     sudo yum install libgomp       # CentOS/RHEL
     ```

## Version Management

- Use conda environments to manage Python versions and dependencies
- Always activate your environment before installing packages:
  ```bash
  conda activate myenv
  ```

## Additional Notes

1. **Docker Users**
   - If using Docker, most dependencies will be handled in the Dockerfile
   - Ensure Docker is installed on your system
   - Check Docker documentation for system-specific installation requirements

2. **Virtual Environment**
   - It's recommended to use a virtual environment for development
   - This project is tested with Python 3.9

3. **Updating Dependencies**
   - Regularly update dependencies for security patches
   - Test thoroughly after any dependency updates
   ```bash
   pip install --upgrade -r requirements.txt
   ```

## Support

If you encounter any dependency-related issues:
1. Check the troubleshooting section above
2. Ensure all system-specific prerequisites are installed
3. Verify you're using the correct Python version (3.9)
4. Create an issue in the project repository with:
   - Your operating system and version
   - Python version
   - Complete error message
   - Steps to reproduce the issue
