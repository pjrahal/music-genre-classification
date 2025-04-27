import importlib
import os
import platform
import shutil
import subprocess
import sys
import venv

from settings import REQUIRED_PACKAGES, ENV_DIR

def run_command(command, sudo=False):
    if sudo and shutil.which("sudo"):
        command = f"sudo {command}"
    print(f"[CMD] {command}")
    subprocess.run(command, shell=True, check=True)

def is_python_installed():
    return shutil.which("python") or shutil.which("python3")

def install_python():
    os_type = platform.system()

    if os_type == "Windows":
        if shutil.which("winget"):
            print("[INFO] Installing Python via winget...")
            run_command("winget install --id Python.Python.3 --source winget")
        else:
            print("[ERROR] winget not found. Please install Python manually from https://www.python.org/downloads/windows/")
            sys.exit(1)

    elif os_type == "Darwin":
        if not shutil.which("brew"):
            print("[INFO] Homebrew not found. Installing Homebrew first...")
            run_command('/bin/bash -c \"$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\"')
        print("[INFO] Installing Python via Homebrew...")
        run_command("brew install python")

    elif os_type == "Linux":
        distro_info = ""
        try:
            with open("/etc/os-release") as f:
                distro_info = f.read()
        except FileNotFoundError:
            pass

        if "Ubuntu" in distro_info or "Debian" in distro_info:
            print("[INFO] Installing Python via apt...")
            run_command("apt update", sudo=True)
            run_command("apt install -y python3 python3-pip", sudo=True)
        else:
            print("[ERROR] Unsupported Linux distribution. Please install Python manually.")
            sys.exit(1)
    else:
        print("[ERROR] Unsupported OS.")
        sys.exit(1)

def check_and_install_python():
    if is_python_installed():
        print("[OK] Python is already installed.")
    else:
        print("[INFO] Python not found. Installing...")
        install_python()
        print("[INFO] ✅ Python installed. Please re-run this script.")
        sys.exit(0)

def setup_virtualenv(env_dir=ENV_DIR):
    if not os.path.isdir(env_dir):
        print(f"[INFO] Creating virtual environment: {env_dir}")
        venv.create(env_dir, with_pip=True)
        print(f"[INFO] ✅ Virtual environment created at '{env_dir}'")
        print("[INFO] To activate it, run:")

        if os.name == "nt":
            print(f"    .\\{env_dir}\\Scripts\\activate")
        else:
            print(f"    source {env_dir}/bin/activate")

        print("\n[INFO] Rerun this script after activating the environment.")
        sys.exit(0)

def install_required_packages():
    for pkg in REQUIRED_PACKAGES:
        module_name = "sklearn" if pkg == "scikit-learn" else pkg
        try:
            importlib.import_module(module_name)
            print(f"[OK] {pkg} already installed.")
        except ImportError:
            print(f"[INFO] Installing {pkg} ...")
            subprocess.run(f"pip install {pkg}", shell=True, check=True)

def write_requirements_file():
    with open("requirements.txt", "w") as f:
        for pkg in REQUIRED_PACKAGES:
            f.write(f"{pkg}\n")
    print("[INFO] requirements.txt written.")

def setup_main():
    check_and_install_python()
    setup_virtualenv()
    print(f"[INFO] Python version: {platform.python_version()}")
    print(f"[INFO] Running inside virtualenv: {sys.prefix}")

    install_required_packages()
    write_requirements_file()
    print("\n✅ Setup complete.")

if __name__ == "__main__":
    setup_main()
