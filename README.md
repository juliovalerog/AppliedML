# Applied Machine Learning

This repository accompanies the Applied Machine Learning course at Albert School. It follows the Telco customer churn case used in Sessions 1 and 2.

## Notebook index

Read the notebooks in numerical order:

1. [`01_telco_basic_eda.ipynb`](notebooks/01_telco_basic_eda.ipynb) — inspect the customers, columns, missing values, target balance, and a few useful relationships.
2. [`02_telco_baselines.ipynb`](notebooks/02_telco_baselines.ipynb) — compare three transparent baselines with confusion matrices, classification metrics, and business costs.
3. [`03_telco_preprocessing_and_cv.ipynb`](notebooks/03_telco_preprocessing_and_cv.ipynb) — build a leakage-safe Logistic Regression Pipeline and evaluate it with five-fold stratified cross-validation.

## Set up your Python environment

A virtual environment is a private Python installation for this repository. It keeps the course packages separate from packages used by other projects. Create it once; on later days you only need to activate it again.

### Before you start: install Python

Python is required. This repository has been tested with Python 3.12.

**Windows**

1. Download Python from the [official Windows page](https://www.python.org/getit/windows/).
2. Run the installer and complete the default Python 3 installation.
3. Close and reopen PowerShell or Command Prompt so it can detect the new installation.
4. Verify it:

```powershell
py -3 --version
```

**macOS**

1. Download and run a Python 3 installer from the [official macOS page](https://www.python.org/downloads/macos/).
2. Close and reopen Terminal.
3. Verify it:

```bash
python3 --version
```

**Ubuntu or Debian Linux**

Install Python, `pip`, and virtual-environment support with the system package manager:

```bash
sudo apt update
sudo apt install python3 python3-pip python3-venv
python3 --version
```

For another Linux distribution, install the equivalent `python3`, `pip`, and `venv` packages with that distribution's package manager.

### Check that pip is available

`pip` installs Python libraries. It normally comes with the official Windows and macOS Python installers.

On Windows:

```powershell
py -m pip --version
```

On macOS or Linux:

```bash
python3 -m pip --version
```

If `pip` is missing from an official Python installation, try:

```bash
python3 -m ensurepip --default-pip
```

On Windows, replace `python3` with `py`. On Linux, prefer installing `python3-pip` with the system package manager as shown above.

### Optional: install Visual Studio Code

JupyterLab is sufficient for this course, so VS Code is optional. To use it:

1. Install [Visual Studio Code](https://code.visualstudio.com/Download).
2. Open VS Code, choose **Extensions**, and install the Microsoft [Python extension](https://marketplace.visualstudio.com/items?itemName=ms-python.python) and [Jupyter extension](https://marketplace.visualstudio.com/items?itemName=ms-toolsai.jupyter).
3. Choose **File → Open Folder** and open the `AppliedML` repository folder.
4. Open **Terminal → New Terminal** and run the environment commands below there.
5. After creating `.venv`, open the Command Palette with `Ctrl+Shift+P` on Windows/Linux or `Cmd+Shift+P` on macOS. Run **Python: Select Interpreter** and choose the interpreter whose path contains `.venv`.
6. When an `.ipynb` notebook is open, use the kernel selector at the top right to confirm that the `.venv` interpreter is selected.

VS Code, its Python extension, and Python itself are separate installations: the editor and extension do not install the Python interpreter.

### 1. Download the repository and open a terminal

If you do not use Git, open the repository page, choose **Code → Download ZIP**, extract the ZIP, and open a terminal in the extracted `AppliedML` folder.

If you use Git, install it from the [official Git downloads page](https://git-scm.com/downloads), then run:

```bash
git clone https://github.com/juliovalerog/AppliedML.git
cd AppliedML
```

The terminal should now be in the folder that contains `README.md` and `requirements.txt`.

Check that Python is available:

```bash
python --version
```

On Windows, use `py --version` if `python` is not recognized. On macOS or Linux, use `python3 --version` if needed.

### 2. Create the virtual environment

On Windows:

```powershell
py -m venv .venv
```

On macOS or Linux:

```bash
python3 -m venv .venv
```

This creates a local `.venv` folder. It is ignored by Git and should not be committed.

### 3. Activate it

In Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

In Windows Command Prompt:

```bat
.venv\Scripts\activate.bat
```

On macOS or Linux:

```bash
source .venv/bin/activate
```

When activation succeeds, the terminal prompt normally begins with `(.venv)`. If PowerShell blocks the activation script, allow it for the current terminal only and try again:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

### 4. Install the course packages

With the environment active, run:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The second command installs every library listed in `requirements.txt`—Jupyter, pandas, NumPy, Matplotlib, seaborn, and scikit-learn—and automatically installs the additional packages they depend on. You do not need to install those dependencies one by one. The first installation can take a few minutes and requires an internet connection.

Check that the installed packages and their dependencies are consistent:

```bash
python -m pip check
```

Then verify that the libraries used in the notebooks can be imported:

```bash
python -c "import pandas, numpy, matplotlib, seaborn, sklearn; print('Environment ready')"
```

If both commands finish without an error, the environment is ready. This installation does not need to be repeated every time you open the project.

### 5. Start Jupyter

Keep the environment active and start Jupyter from the repository root:

```bash
python -m jupyter lab
```

Open the `notebooks` folder and read the notebooks in numerical order. If Jupyter asks you to choose a kernel, select the Python kernel associated with `.venv`.

### 6. Finish or return later

To leave the environment:

```bash
deactivate
```

The next time you work on the course, return to the repository, repeat the activation command from step 3, and start Jupyter again. You do not need to recreate `.venv`.

## Dataset

The course copy of the IBM Telco Customer Churn sample is included at [`data/telco.csv`](data/telco.csv). It contains 7,043 customer records and 21 columns. Public references are available from the [IBM sample repository](https://github.com/IBM/telco-customer-churn-on-icp4d/tree/master/data) and the [Kaggle dataset page](https://www.kaggle.com/datasets/blastchar/telco-customer-churn).
