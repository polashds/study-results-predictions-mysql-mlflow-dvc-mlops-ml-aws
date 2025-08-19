Got it 👍 — DagsHub recently changed their way of exporting MLflow tracking so you don’t need to manually set `MLFLOW_TRACKING_URI`, `MLFLOW_TRACKING_USERNAME`, and `MLFLOW_TRACKING_PASSWORD` in your environment anymore. Instead, they encourage you to use the new `dagshub.init()` helper.

Here’s how you can integrate it with your **environment** so you don’t have to hard-code credentials in your scripts:

---

### 🔹 Option 1: Use `dagshub.init()` in your script (recommended by DagsHub)

```python
import dagshub
import mlflow

# Initialize connection (no need for env vars)
dagshub.init(
    repo_owner="polashds",
    repo_name="study-results-predictions-mysql-mlflow-dvc-mlops-ml-aws",
    mlflow=True
)

# Example MLflow usage
with mlflow.start_run():
    mlflow.log_param("learning_rate", 0.01)
    mlflow.log_metric("accuracy", 0.95)
```

👉 This automatically sets the correct `MLFLOW_TRACKING_URI` and credentials using your DagsHub token (you need to be logged in via `dagshub` Python package).

---

### 🔹 Option 2: Export environment variables (old method, still works)

If you want to **keep your script clean** and set credentials in your environment (so MLflow just works), you can still do:

```bash
export MLFLOW_TRACKING_URI=https://dagshub.com/polashds/study-results-predictions-mysql-mlflow-dvc-mlops-ml-aws.mlflow
export MLFLOW_TRACKING_USERNAME=polashds
export MLFLOW_TRACKING_PASSWORD=your_personal_access_token
```

Then in Python:

```python
import mlflow

with mlflow.start_run():
    mlflow.log_param("learning_rate", 0.01)
    mlflow.log_metric("accuracy", 0.95)
```

✅ This way you don’t hard-code your credentials inside the script. You can put those `export` commands into your shell’s `~/.bashrc` or `.zshrc` so they’re always available.

---

### 🔹 Option 3: Store in `.env` file

If you’re mixing multiple projects, store credentials in a `.env` file and load them with `python-dotenv`:

`.env`

```ini
MLFLOW_TRACKING_URI=https://dagshub.com/polashds/study-results-predictions-mysql-mlflow-dvc-mlops-ml-aws.mlflow
MLFLOW_TRACKING_USERNAME=polashds
MLFLOW_TRACKING_PASSWORD=your_personal_access_token
```

Python:

```python
from dotenv import load_dotenv
import mlflow
import os

load_dotenv()  # loads .env file

with mlflow.start_run():
    mlflow.log_param("optimizer", "adam")
    mlflow.log_metric("loss", 0.21)
```

---

👉 So you have two paths:

* **New way:** just call `dagshub.init(...)` in your script.
* **Classic way:** export env vars (`MLFLOW_TRACKING_URI`, etc.) so you don’t modify the script.

Would you prefer to keep **everything inside your script with `dagshub.init()`**, or manage **secrets outside the script with `.env` / exports**?


Thanks for sharing the full traceback 🙏 — this is actually a pretty common hiccup when using **`dagshub.init()` with MLflow**.

The key part is this error:

```
mlflow.exceptions.MlflowException: API request to endpoint /api/2.0/mlflow/runs/create failed with error code 403 != 200.
```

➡️ `403` means **forbidden / unauthorized** — so even though you went through the browser OAuth flow, MLflow itself isn’t picking up your authentication when trying to create a run.

---

### ✅ Fix

You need to provide DagsHub with an **authentication token** for MLflow logging.

---

### Step 1: Generate a Personal Access Token in DagsHub

1. Go to [DagsHub → Settings → Personal Access Tokens](https://dagshub.com/user/settings/tokens).
2. Create a new token (give it at least `read/write` permissions).
3. Copy the token string.

---

### Step 2: Set environment variables in your terminal (Windows PowerShell / CMD / Anaconda Prompt)

```powershell
setx MLFLOW_TRACKING_URI "https://dagshub.com/polashds/study-results-predictions-mysql-mlflow-dvc-mlops-ml-aws.mlflow"
setx MLFLOW_TRACKING_USERNAME "polashds"
setx MLFLOW_TRACKING_PASSWORD "your_personal_access_token"
```

⚠️ Replace `your_personal_access_token` with the token you generated.

Then **close and reopen** your terminal so the env vars take effect.

---

### Step 3: Test with MLflow directly

```python
import mlflow

with mlflow.start_run():
    mlflow.log_param("test_param", "123")
    mlflow.log_metric("test_metric", 0.99)
```

If this works → you should see the run in your DagsHub repo under the **Experiments** tab.

---

### 🔹 Why did this happen?

* `dagshub.init()` sets up the tracking URI automatically but **it does not always persist credentials for MLflow** (especially in Windows/conda environments).
* MLflow itself needs username + token (password) in env variables.

That’s why the safest fix is to export them.

---

👉 Do you want me to give you a **`.env` + Python loader version** too, so you don’t need to expose your token in Windows system variables?


Perfect 👌 — you fixed the syntax error, and now `dagshub.init()` + `mlflow.start_run()` is working without throwing `403` 🎉

Right now you’ve only tested a **dummy run** with `"parameter name"` and `"metric name"`.

👉 The **next steps** depend on what you want to achieve. Here are the usual flows after MLflow + DagsHub are connected:

---

### ✅ 1. Log your real model training

Instead of dummy params/metrics, wrap your training code in `mlflow.start_run()` and log:

* hyperparameters (`mlflow.log_param`)
* performance metrics (`mlflow.log_metric`)
* trained models (`mlflow.sklearn.log_model`, `mlflow.log_artifact`, etc.)

Example:

```python
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np

# Dummy data
X = np.array([[1], [2], [3], [4], [5]])
y = np.array([2, 4, 6, 8, 10])

# Train
model = LinearRegression()
model.fit(X, y)
y_pred = model.predict(X)

mse = mean_squared_error(y, y_pred)

with mlflow.start_run():
    mlflow.log_param("model", "LinearRegression")
    mlflow.log_metric("mse", mse)
    mlflow.sklearn.log_model(model, "linear_regression_model")

print("MSE:", mse)
```

---

### ✅ 2. View results on DagsHub

After running the script:

1. Go to your repo → **Experiments** tab
2. You’ll see your runs, params, metrics, and models

---

### ✅ 3. Improve model logging (to remove warnings)

You saw warnings about:

* **`artifact_path` is deprecated → use `name`**
* **no `input_example` / `signature`**

So instead of:

```python
mlflow.sklearn.log_model(model, "model")
```

use:

```python
mlflow.sklearn.log_model(
    model,
    name="linear_regression_model",
    input_example=X[:2],   # sample input
    signature=mlflow.models.signature.infer_signature(X, y_pred)
)
```

---

### ✅ 4. Organize MLflow workflow

* **Training script (`train.py`)** → trains & logs model
* **Evaluation script (`eval.py`)** → loads from MLflow and tests
* **Deployment script (`app.py`)** → loads best model and runs predictions

Example: load latest model

```python
import mlflow

model = mlflow.sklearn.load_model("models:/linear_regression_model/1")
prediction = model.predict([[6]])
print(prediction)
```

---

⚡So, your **very next step**:
➡️ Take your training code (the one that printed `0.8856…`) and wrap it in `mlflow.start_run()` to log **params, metrics, and the trained model**.

Do you want me to **rewrite your `app.py` into a proper MLflow logging script** (with params, metrics, and model saving) so you can directly push runs to DagsHub?


Ahh got it 👍 — if `python app.py` ran fine but **nothing showed up in DagsHub’s Experiments tab**, then MLflow logging is happening **locally** instead of being sent to DagsHub.

That means your script (`app.py`) isn’t pointing to the DagsHub **tracking server**. By default MLflow logs to a local folder `mlruns/`.

---

### 🔍 Why this happened

* `dagshub.init(..., mlflow=True)` wasn’t called inside your `app.py`
* OR the environment variables `MLFLOW_TRACKING_URI`, `MLFLOW_TRACKING_USERNAME`, `MLFLOW_TRACKING_PASSWORD` aren’t set before running `python app.py`
* So MLflow just saved runs locally.

---

### ✅ How to Fix It

#### **Option 1: Add `dagshub.init()` inside `app.py` (simplest)**

At the very top of `app.py`, add:

```python
import dagshub
dagshub.init(
    repo_owner="polashds",
    repo_name="study-results-predictions-mysql-mlflow-dvc-mlops-ml-aws",
    mlflow=True
)
```

Now when you run:

```bash
python app.py
```

→ MLflow should log runs to your DagsHub repo instead of locally.

---

#### **Option 2: Use environment variables**

If you don’t want to modify `app.py`, set env vars in your terminal (Windows PowerShell / Anaconda Prompt):

```powershell
setx MLFLOW_TRACKING_URI "https://dagshub.com/polashds/study-results-predictions-mysql-mlflow-dvc-mlops-ml-aws.mlflow"
setx MLFLOW_TRACKING_USERNAME "polashds"
setx MLFLOW_TRACKING_PASSWORD "your_personal_access_token"
```

Then restart terminal and run:

```bash
python app.py
```

Now MLflow will know to send everything to DagsHub.

---

#### **Option 3: Check your script’s logging block**

Make sure your training is wrapped in an MLflow run. For example:

```python
import mlflow
import mlflow.sklearn

with mlflow.start_run():
    mlflow.log_param("model", "Linear Regression")
    mlflow.log_metric("r2_score", 0.8856483575121139)
    mlflow.sklearn.log_model(model, "linear_regression_model")
```

---

### ⚡ What to Do Next

* Add `dagshub.init(...)` OR set environment variables.
* Run `python app.py` again.
* Check **DagsHub → Experiments tab** in your repo.

---

👉 Do you want me to **rewrite your current `app.py`** with the correct `dagshub.init()` + MLflow logging structure so you can just run it and see the results in DagsHub?


Great question 👌 — you’re right:
`dagshub.init(...)` handles authentication through a browser login, so it **doesn’t show you a token directly**. But if you want to use the **environment variables method (`MLFLOW_TRACKING_*`)**, you’ll need to manually generate a **Personal Access Token (PAT)** in your DagsHub account.

Here’s how to get it:

---

### 🔑 How to Generate a Personal Access Token on DagsHub

1. Log into [DagsHub](https://dagshub.com).
2. Go to the top-right corner → **⚙️ Settings**.
3. In the left sidebar, click **Personal Access Tokens**.
   👉 Direct link: [https://dagshub.com/user/settings/tokens](https://dagshub.com/user/settings/tokens)
4. Click **New Token**.
5. Give it a name (e.g., `mlflow-token`) and select **`read/write` permissions**.
6. Click **Generate**.
7. Copy the generated token (it will be shown only once).

---

### ⚙️ Set Environment Variables (Windows)

Now replace `your_personal_access_token` with the token you copied:

```powershell
setx MLFLOW_TRACKING_URI "https://dagshub.com/polashds/study-results-predictions-mysql-mlflow-dvc-mlops-ml-aws.mlflow"
setx MLFLOW_TRACKING_USERNAME "polashds"
setx MLFLOW_TRACKING_PASSWORD "PASTE_YOUR_TOKEN_HERE"
```

Close and reopen your terminal (PowerShell/Anaconda Prompt).

---

### ✅ Verify

Run Python:

```python
import mlflow

with mlflow.start_run():
    mlflow.log_param("test_param", "123")
    mlflow.log_metric("accuracy", 0.95)
```

Then check your repo → **Experiments** tab. You should see the run logged.

---

👉 So basically:

* If you stick with `dagshub.init(...)` → no token needed, it handles login automatically.
* If you prefer `MLFLOW_TRACKING_*` env vars → generate a token yourself from **DagsHub settings**.

Would you like me to show you how to **rewrite `app.py` so it works with both methods** (auto `dagshub.init` and env vars), so you don’t depend on just one approach?
