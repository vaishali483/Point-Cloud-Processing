import os
import platform
import sys
import re
import requests
import argparse
from pathlib import Path
import subprocess
import math
import shutil

os.environ["PYTHONIOENCODING"] = "utf-8"

ZED_SDK_MAJOR = ""
ZED_SDK_MINOR = ""

PYTHON_MAJOR = ""
PYTHON_MINOR = ""

OS_VERSION = ""
ARCH_VERSION = platform.machine()

whl_platform_str = ""

base_URL = "https://download.stereolabs.com/zedsdk/"

def pip_install(package, force_install=False, ignore_install=False, upgrade=False, break_system_packages=False):
    try:
        # Set the environment variable
        if break_system_packages:
            os.environ['PIP_BREAK_SYSTEM_PACKAGES'] = '1'

        call_list = [sys.executable, "-m", "pip", "install"]
        if ignore_install:
            call_list.append("--ignore-installed")
        if force_install:
            call_list.append("--force-reinstall")
        if upgrade:
            call_list.append("--upgrade")
        if break_system_packages:
            call_list.append("--break-system-packages")
        call_list.append(package)
        err = subprocess.check_call(call_list)
    except Exception as e:
        err = 1
    return err


def check_valid_file(file_path):
    """
    Checks if the specified file is likely a valid .whl (Python wheel) file.
    - Wheel files are ZIP archives, so their first 4 bytes are always b'PK\x03\x04'.
    - This function checks that the file exists, is at least 150 KB, and starts with the ZIP magic bytes.
    - This helps avoid mistaking an HTML error page for a real wheel file after download.

    Args:
        file_path (str): Path to the file to check.

    Returns:
        bool: True if the file looks like a valid wheel, False otherwise.
    """
    try:
        file_size = os.stat(file_path).st_size / 1000.
    except FileNotFoundError:
        file_size = 0

    # .whl files are ZIP archives, which always start with b'PK\x03\x04'
    try:
        with open(file_path, 'rb') as f:
            header = f.read(4)
        is_zip = (header == b'PK\x03\x04')
    except Exception:
        is_zip = False

    # Size > 150 KB and is a ZIP archive (likely a valid wheel)
    return (file_size > 150) and is_zip


def install_win_dep(name, py_vers, break_system_packages=False):
    if py_vers < 310:
        package_vers = "3.1.5"
    elif py_vers >= 310 and py_vers < 312:
        package_vers = "3.1.6"
    else:
        package_vers = "3.1.9"

    whl_file = f"{name}-{package_vers}-cp{py_vers}-cp{py_vers}"
    if py_vers <= 37:
        whl_file = whl_file + "m"
    whl_file = whl_file + "-win_amd64.whl"

    whl_file_URL = "https://download.stereolabs.com/py/" + whl_file
    print("-> Downloading " + whl_file)
    whl_file = os.path.join(dirname, whl_file)
    r = requests.get(whl_file_URL, allow_redirects=True)
    open(whl_file, 'wb').write(r.content)
    pip_install(whl_file, break_system_packages=break_system_packages)


def get_pyzed_directory():
    try:
        call_list = [sys.executable, "-m", "pip", "show", "pyzed"]
        lines = subprocess.check_output(call_list, encoding="utf-8", errors="replace").splitlines()
        for line in lines:
            key_word = "Location:"
            if line.startswith(key_word):
                directory = line[len(key_word):].strip()
                if os.path.isdir(directory):
                    print("Pyzed directory is " + directory)
                    return directory + "/pyzed"
                else:
                    print("ERROR : '" + directory + "' is not a directory")

        print("ERROR : Unable to find pyzed installation folder")
        print(lines)

    except Exception as e:
        print("ERROR : Unable to find pyzed installation folder.")
        return ""


def check_zed_sdk_version_private(file_path):
    global ZED_SDK_MAJOR
    global ZED_SDK_MINOR

    with open(file_path, "r", encoding="utf-8") as myfile:
        data = myfile.read()

    p = re.compile("ZED_SDK_MAJOR_VERSION (.*)")
    ZED_SDK_MAJOR = p.search(data).group(1)

    p = re.compile("ZED_SDK_MINOR_VERSION (.*)")
    ZED_SDK_MINOR = p.search(data).group(1)


def check_zed_sdk_version(file_path):
    file_path_ = file_path + "/sl/Camera.hpp"
    try:
        check_zed_sdk_version_private(file_path_)
    except AttributeError:
        file_path_ = file_path + "/sl_zed/defines.hpp"
        check_zed_sdk_version_private(file_path_)

def can_write_to_dir(dirname):
    if not (os.path.exists(dirname) and os.path.isdir(dirname) and os.access(dirname, os.W_OK)):
        return False
    # Try to actually write because Windows permissions can be unreliable
    try:
        testfile = os.path.join(dirname, 'temp_test_file.tmp')
        with open(testfile, 'w') as f:
            f.write('test')
        os.remove(testfile)
        return True
    except Exception as e:
        return False
    
parser = argparse.ArgumentParser(description='Helper script to download and setup the ZED Python API')
parser.add_argument('--path', help='whl file destination path')
parser.add_argument('--force', action='store_true', help='Force install and use --break-system-packages (pip >= 23.0)')
args = parser.parse_args()

arch = platform.architecture()[0]
if arch != "64bit":
    print("ERROR : Python 64bit must be used, found " + str(arch))
    sys.exit(1)

# If path empty, take pwd
dirname = args.path or os.getcwd()

# If no write access, download in home
if not can_write_to_dir(dirname):
    dirname = str(Path.home())

print("-> Downloading to '" + str(dirname) + "'")

if sys.platform == "win32":
    zed_path = os.getenv("ZED_SDK_ROOT_DIR")
    if zed_path is None:
        print("Error: you must install the ZED SDK.")
        sys.exit(1)
    else:
        check_zed_sdk_version(zed_path + "/include")
    OS_VERSION = "win" + "_" + str(ARCH_VERSION).lower()
    whl_platform_str = "win"

elif "linux" in sys.platform:
    if "aarch64" in ARCH_VERSION:
        OS_VERSION = "linux_aarch64"
    else:
        OS_VERSION = "linux_x86_64"

    zed_path = "/usr/local/zed"
    if not os.path.isdir(zed_path):
        print("Error: you must install the ZED SDK.")
        sys.exit(1)
    check_zed_sdk_version(zed_path + "/include")
    whl_platform_str = "linux"
else:
    print("Unknown system.platform: %s  Installation failed, see setup.py." % sys.platform)
    sys.exit(1)

PYTHON_MAJOR = platform.python_version().split(".")[0]
PYTHON_MINOR = platform.python_version().split(".")[1]

whl_python_version = "-cp" + str(PYTHON_MAJOR) + str(PYTHON_MINOR) + "-cp" + str(PYTHON_MAJOR) + str(PYTHON_MINOR)
if int(PYTHON_MINOR) < 8:
    whl_python_version += "m"

disp_str = "Detected platform: \n\t " + str(OS_VERSION) + "\n\t Python " + str(PYTHON_MAJOR) + "." + str(PYTHON_MINOR)
disp_str += "\n\t ZED SDK " + str(ZED_SDK_MAJOR) + "." + str(ZED_SDK_MINOR)
print(disp_str)

whl_file = "pyzed-" + str(ZED_SDK_MAJOR) + "." + str(
    ZED_SDK_MINOR) + whl_python_version + "-" + whl_platform_str + "_" + str(ARCH_VERSION).lower() + ".whl"

whl_file_URL = base_URL + str(ZED_SDK_MAJOR) + "." + str(ZED_SDK_MINOR) + "/whl/" + OS_VERSION + "/" + whl_file
whl_file = os.path.join(dirname, whl_file)

print("-> Checking if " + whl_file_URL + " exists and is available")
try:
    r = requests.get(whl_file_URL, allow_redirects=True)
    open(whl_file, 'wb').write(r.content)
except PermissionError as e:
    print("Permission denied: Unable to write the packages. Run this script as admin or copy it in a different path and retry.")
    sys.exit(1)
except requests.exceptions.HTTPError as e:
    print("Error downloading whl file ({})".format(e))
except requests.exceptions.URLError as e:
    print("Invalid SSL certificate, trying to fix the issue by reinstalling 'certifi' package")
    err = pip_install("certifi", force_install=True, upgrade=True, break_system_packages=args.force)
    if err == 0:
        # Retrying
        try:
            r = requests.get(whl_file_URL, allow_redirects=True)
            open(whl_file, 'wb').write(r.content)
        except Exception as e:
            print("Error downloading whl file ({})".format(e))


if check_valid_file(whl_file):
    # Internet is ok, file has been downloaded and is valid
    print("-> Found ! Downloading python package into " + whl_file)

    print("-> Installing necessary dependencies")
    err = 0
    if "aarch64" in ARCH_VERSION:
        # On jetson numpy is built from source and need other packages
        err_wheel = pip_install("wheel", break_system_packages=args.force)
        err_cython = pip_install("cython", break_system_packages=args.force)
        err = err_wheel + err_cython
    err_numpy = pip_install("numpy", break_system_packages=args.force)

    if err != 0 or err_numpy != 0:
        print("ERROR : An error occurred, 'pip' failed to setup python dependencies packages (pyzed was NOT correctly setup)")
        sys.exit(1)

    err_pyzed = pip_install(whl_file, force_install=True, break_system_packages=args.force)
    if err_pyzed == 0:
        print("Done")
    else:
        print("ERROR : An error occurred, 'pip' failed to setup pyzed package (pyzed was NOT correctly setup)")
        sys.exit(1)

    if sys.platform == "win32":
        print("Installing OpenGL dependencies required to run the samples")
        py_vers = int(int(PYTHON_MAJOR) * math.pow(10, len(PYTHON_MINOR)) + int(PYTHON_MINOR))
        install_win_dep("PyOpenGL", py_vers, break_system_packages=args.force)
        install_win_dep("PyOpenGL_accelerate", py_vers, break_system_packages=args.force)

        # Two files must be copied into pyzed folder : sl_zed64.dll and sl_ai64.dll
        pyzed_dir = get_pyzed_directory()
        source_dir = zed_path + "/bin"
        files = ["/sl_ai64.dll", "/sl_zed64.dll"]

        for file in files:
            if os.path.isfile(source_dir + file):
                shutil.copy(source_dir + file, pyzed_dir + file)
            else:
                print("ERROR : An error occurred, 'pip' failed to copy dll file " + source_dir + file + " (pyzed was NOT correctly setup)")
    else: # Only on linux, on windows this script should be used every time to avoid library search path issues
        print("  To install it later or on a different environment run : \n python -m pip install --ignore-installed " + whl_file)
    sys.exit(0)
else:
    print("\nUnsupported platforms, no pyzed file available for this configuration\n It can be manually installed from "
        "source https://github.com/stereolabs/zed-python-api")
    sys.exit(1)