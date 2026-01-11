import subprocess
import sys

print("Installing dependencies...")

print("numpy: pip install numpy")
subprocess.check_call([sys.executable, "-m", "pip", "install", "numpy"])

print("pygame: pip install pygame")
subprocess.check_call([sys.executable, "-m", "pip", "install", "pygame"])

print("matplotlib: pip install matplotlib")
subprocess.check_call([sys.executable, "-m", "pip", "install", "matplotlib"])

