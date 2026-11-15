# Applied Machine Learning

This repository accompanies the Applied Machine Learning course at Albert School. It follows the Telco customer churn case used in Sessions 1–7, with a first local FastAPI example for Session 8. **Session 8 is a preview of next semester's dedicated MLOps & Model Deployment course.**

## Notebook index

Read the notebooks in numerical order:

1. [`01_telco_basic_eda.ipynb`](notebooks/01_telco_basic_eda.ipynb) — inspect the customers, columns, missing values, target balance, and a few useful relationships.
2. [`02_telco_baselines.ipynb`](notebooks/02_telco_baselines.ipynb) — compare three transparent baselines with confusion matrices, classification metrics, and business costs.
3. [`03_telco_preprocessing_and_cv.ipynb`](notebooks/03_telco_preprocessing_and_cv.ipynb) — build a leakage-safe Logistic Regression Pipeline and evaluate it with five-fold stratified cross-validation.
4. [`04_telco_trees_and_forest.ipynb`](notebooks/04_telco_trees_and_forest.ipynb) — change only the model: compare a small tree, a full tree, and a Random Forest with the Logistic Regression reference.
5. [`05_telco_boosting_and_model_choice.ipynb`](notebooks/05_telco_boosting_and_model_choice.ipynb) — add boosting, compare five models, and choose a candidate using evidence and business priorities.
6. [`06_telco_feature_engineering.ipynb`](notebooks/06_telco_feature_engineering.ipynb) — Session 4: test feature hypotheses, categorical encodings, and complete solutions with the same development folds.
7. [`07_telco_comparison_and_selection.ipynb`](notebooks/07_telco_comparison_and_selection.ipynb) — Session 5: paired AP differences, illustrative Wilcoxon, paired bootstrap, multiple comparisons, and an executed nested-CV selection procedure.
8. [`08_telco_economic_uncertainty.ipynb`](notebooks/08_telco_economic_uncertainty.ipynb) — Session 5: reproduce the training and out-of-fold economic comparison of Logistic Regression with original inputs versus added logs, including 5,000 paired bootstrap draws.
9. [`09_telco_probability_calibration.ipynb`](notebooks/09_telco_probability_calibration.ipynb) — Session 6: distinguish ranking, classification and probabilities; compare raw, sigmoid and isotonic predictions with Brier, learned mappings, actual bin support and Wilson intervals.
10. [`10_telco_customer_group_diagnostics.ipynb`](notebooks/10_telco_customer_group_diagnostics.ipynb) — Session 6: inspect calibration and cutoff-dependent detection for monthly and longer contracts, with uncertainty and group-specific denominators.
11. [`11_telco_thresholds_and_business_decisions.ipynb`](notebooks/11_telco_thresholds_and_business_decisions.ipynb) — Session 6: select uncapped policies on validation, evaluate on holdout, isolate a contact cap, bootstrap paired value differences, explore economic sensitivity and design a policy A/B test.
12. [`12_telco_global_importance.ipynb`](notebooks/12_telco_global_importance.ipynb) — Session 7: explain global reliance with validation permutation, compare XGBoost training gain on the same model, and investigate correlated billing inputs with standalone, within-decile and joint shuffles.
13. [`13_telco_local_explanations.ipynb`](notebooks/13_telco_local_explanations.ipynb) — Session 7: reconstruct individual log-odds scores with linear SHAP, contrast two class customers, inspect built-in TreeSHAP and global summaries, and explain the policy to different audiences; optional PDP/ICE syllabus bridge.
14. [`14_telco_local_fastapi.ipynb`](notebooks/14_telco_local_fastapi.ipynb) — Session 8 preview: save and reload the Session 6 pipeline, start a local FastAPI service, send a complete JSON customer and inspect prediction and validation responses. Follow the [terminal walkthrough](serving/README.md) alongside it.

Each notebook runs independently from the `notebooks` directory and includes saved outputs. Notebook 14 additionally needs the serving dependencies and a local Uvicorn process in a separate terminal, as its instructions explain. Sessions 4–5 use only development data: the stratified 20% final test from notebook 03 (`random_state=42`) remains closed. Development comparisons reuse five stratified, shuffled folds with seed 42. Notebook 07 runs nested CV within development; its outer results assess the restricted selection procedure, with the prior classroom search explicitly acknowledged.

The economic calculation reproduces approximately +€127,796 per million customers on training predictions and +€234,292 on out-of-fold predictions, with an exploratory 95% interval from −€149,627 to +€575,080. Its central estimate is positive, but the interval includes losses. Predictions remain fixed during bootstrap; costs and retention effectiveness are assumptions, not measured causal effects.

Sessions 4–5 require scikit-learn 1.9 or later for explicit cross-validation splitters in target encoding. SciPy is listed directly for Wilcoxon (it was already installed through scikit-learn). Notebooks 06–14 print the library versions used for their saved outputs.

The first eleven notebooks have been reviewed against notebook 01's teaching style: comments explain each executable step, intermediate values have readable names, and calculations are separated from their plots. Straightforward loops replace dense nested comprehensions. Row-wise feature formulas use ordinary functions; the frequency encoder keeps a small fitted class because it must remember training statistics. Repeated fold evaluation and accounting retain local helpers with their steps exposed. These modelling notebooks remain self-contained, with no shared helper module required. The readability review preserves the experiments, splits, seeds, model settings and economic assumptions; all eleven were checked from clean kernels. The new Session 7 notebooks follow the same conventions. Session 8 deliberately moves its transform into an importable module so the saved pipeline works in a separate process.

### Session 6 protocol and economic assumptions

Continue in numerical order: notebook 09 introduces probability diagnostics, 10 explores contract groups, and 11 fixes and evaluates business policies. Each rebuilds its experiment independently. These notebooks follow the local Session 6 v4 HTML, experiment script, evidence JSON and storytelling notes; those instructor files are read-only references and are not needed to execute the repository.

The dataset SHA-256 is checked against the classroom evidence. The stratified split (seed 42) contains 5,634 development and 1,409 holdout customers. Development is split into 3,943 fitting and 1,691 validation customers (seed 606). Sigmoid and isotonic calibration use five shuffled stratified fitting folds, seed 606, with `ensemble=False`. Validation selects the correction and each model's cutoff from 0.1–1.0; refitting uses development with decisions fixed. Session 6 introduces holdout evaluation, but that holdout has already been inspected in classroom materials: this is a teaching reproduction, not a fresh untouched confirmation. Group diagnostics are exploratory.

Session 6 compares original-input Logistic Regression with Logistic Regression plus two logs, distinct from notebook 07's algorithm comparison. Relative to no offers (zero incremental contribution), it assumes €300 preserved margin, a €20 discount paid only when accepted with staying, 20% retention of would-be churners, 50% discount acceptance among would-be stayers, and zero contact and implementation costs by default. Thus expected contribution is `56 × TP − 10 × FP`. Notebook 08 instead uses 40% retention, €22 per contact and fixed top-20% capacity; its estimates and intervals must not be carried over. Session 6's main policy has no contact cap. A separate comparison holds model and cutoff fixed while adding a 20% cap, rounded up with stable score ties resolved by holdout row order.

Campaign counts, contributions, differences and interval bounds are reported per million customers using the actual validation, holdout or group denominator. Actual sample sizes and bin support remain visible. Scaling does not increase precision. The 4,000-draw paired bootstrap (seed 606) holds models, cutoffs and intervention assumptions fixed; it omits full retraining/selection and causal offer-effect uncertainty. Sensitivity scenarios are separate from confidence intervals. A randomized policy experiment is needed to measure intervention value.

The v4 reference used scikit-learn 1.8.0; saved notebooks use the compatible repository environment (versions printed in each notebook), preserving the >=1.9 requirement. Notebook 11 calculates differences from the classroom figures instead of replacing computed values with reference constants.

Executed with Python 3.12.9, scikit-learn 1.9.1, NumPy 2.5.3 and pandas 3.0.6, all three Session 6 notebooks pass from clean kernels. Raw finalist probabilities win validation Brier; both models select cutoff 0.2. Holdout expected contributions reproduce v4 at €10,127,750.18 and €10,205,819.73 per million, a paired difference of +€78,069.55 with conditional 95% interval [−€231,405.25, +€401,703.34]. Group detection checks also match. There is no material discrepancy in these campaign results despite the environment change; small continuous probability differences need not alter cutoff decisions. The interval includes losses, and offer-response assumptions still require experimental validation.

### Session 7: explain the model and the decision

Notebooks 12–13 follow the local Session 7 v3 HTML and its embedded experiment and evidence: global permutation, tree gain, correlated inputs, local SHAP, global SHAP and the audience question. The syllabus also mentions PDPs; notebook 13 includes an optional PDP/ICE exercise after the main SHAP sequence, exposing unsupported tenure/charge combinations. Instructor sources remain read-only and are not required at runtime.

Both notebooks rebuild the Session 6 fitting/validation experiment independently, without using holdout. The selected raw Logistic Regression finalist retains original inputs plus two logs. Raw-column shuffles recompute derived features; 20 repeats use validation AP. Within-decile shuffles approximately condition on one related variable, while joint shuffling preserves the tenure/total-charge pairing. These diagnose model reliance, not causal effects. The comparison XGBoost uses the raw-input classroom recipe with 100 trees, learning rate 0.1, depth 2 and seed 42.

Linear SHAP is computed directly as `coefficient × (transformed input − fitting mean)` in log odds, grouped back to raw fields. Base plus contributions reconstruct every validation score and probability. This uses an interventional background convention; XGBoost's built-in TreeSHAP comparison uses its tree-path-dependent default. Global mean absolute SHAP measures score attribution, while permutation measures AP loss. Neither establishes which retention action would work. No new interpretability package is needed.

### Session 8 preview: serve a local model

Read [notebook 14](notebooks/14_telco_local_fastapi.ipynb) and the [complete local walkthrough](serving/README.md). After activating `.venv`, run these commands **from the repository root**:

```bash
python -m pip install -r serving/requirements.txt
python -m pip check
python -m serving.train
python -m uvicorn serving.app:app --host 127.0.0.1 --port 8000
```

Keep that server running. In a second terminal at the root, activate the same environment and run:

```bash
python -m serving.client
python -m serving.check_api
```

The API loads the complete fitted Session 6 pipeline once per server process. The full input is in [`serving/customer.json`](serving/customer.json); the response is approximately `{"churn_probability": 0.3598182, "model_version": "telco-s6-v1"}`. This is a risk estimate. The caller separately applies its offer cutoff and economics. The HTTP check compares the response with direct saved-pipeline scoring, checks required-field/type/category errors and verifies fitted imputation for an explicit null total charge. Stop the server with `Ctrl+C` after the exercise.

FastAPI, Pydantic and Uvicorn are optional serving dependencies; joblib is declared directly there. Generated model artifacts and environment metadata live in ignored `serving/artifacts/`. This first example ends at local HTTP scoring. The lesson briefly bridges to input/concept drift and delayed outcome checks; production infrastructure, automation and operational monitoring are reserved for next semester's dedicated course.

Notebooks 12–14 were executed from clean kernels with Python 3.12.9, scikit-learn 1.9.1, pandas 3.0.6, NumPy 2.5.3 and XGBoost 3.4.1. The local serving walkthrough was checked with FastAPI 0.163.1, Pydantic 2.14.3, Uvicorn 0.54.0 and joblib 1.6.0 on Windows PowerShell. Dependency checks passed; the Python client and `curl.exe` returned HTTP 200 with risk 0.3598181970496265. Real HTTP checks also passed for missing `Contract` (422), invalid numeric type (422), unknown contract category (422) and null total-charge imputation (200). The macOS/Linux shell examples are provided but were not run on those operating systems.

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

The second command installs every library listed in `requirements.txt`—Jupyter, pandas, NumPy, Matplotlib, seaborn, scikit-learn, XGBoost, and LightGBM—and automatically installs the additional packages they depend on. You do not need to install those dependencies one by one. The first installation can take a few minutes and requires an internet connection.

Check that the installed packages and their dependencies are consistent:

```bash
python -m pip check
```

Then verify that the libraries used in the notebooks can be imported:

```bash
python -c "import pandas, numpy, matplotlib, seaborn, sklearn, xgboost, lightgbm; print('Environment ready')"
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
