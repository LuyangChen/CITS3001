Super Mario Rule-Based Agent Setup Guide

This guide provides a step-by-step procedure to set up a rule-based agent for Super Mario in a conda virtual environment.

Prerequisites
Anaconda distribution for Python.
Visual Studio Code (VSCode).

Setup Instructions
1. Install Anaconda
Download the appropriate Anaconda installer for your operating system from the official Anaconda website.

2. Install Visual Studio Code (VSCode)
If you haven't installed VSCode yet, download it from the official website and follow the installation instructions.

3. Create a New Conda Environment
Open your terminal or command prompt and run the following commands:

# Create a new conda environment named 'mario' with Python 3.8
conda create -n mario python=3.8

# Activate the 'mario' environment
conda activate mario

4. Open Your Project in VSCode
Navigate to your project folder and open it in VSCode.

5. Select Python Interpreter
In VSCode:

Press Ctrl + Shift + P to open the command palette.
Type and select "Python: Select Interpreter".
From the list, choose the interpreter that corresponds to the mario conda environment.

6. Run the Script
Open the mario_locate_objects.py file in VSCode and run it.

