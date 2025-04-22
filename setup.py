import importlib
import os
import platform
import sys
import subprocess
import venv

from settings import REQUIRED_PACKAGES, ENV_DIR

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
    setup_virtualenv()
    print(f"[INFO] Python version: {platform.python_version()}")
    print(f"[INFO] Running inside virtualenv: {sys.prefix}")

    install_required_packages()
    write_requirements_file()
    print("\n✅ Setup complete.")

if __name__ == "__main__":
    setup_main()
