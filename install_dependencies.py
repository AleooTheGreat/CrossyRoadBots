import subprocess
import sys

print("Installing dependencies...")

print("pygame: pip install pygame")
subprocess.check_call([sys.executable, "-m", "pip", "install", "pygame"])

print("matplotlib: pip install matplotlib")
subprocess.check_call([sys.executable, "-m", "pip", "install", "matplotlib"])

