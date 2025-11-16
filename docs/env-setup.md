# Environment Setup (Windows + PowerShell)

This project targets Python 3.10 with TensorFlow/Keras 2.15.x.

## Option A: Using pyenv-win (preferred)

1. Ensure pyenv-win is up-to-date:
   - If `pyenv update` fails, update pyenv-win via its installer or Git pull per docs: <https://github.com/pyenv-win/pyenv-win>
2. Install and set local Python:

   ```powershell
   pyenv install 3.10.13
   pyenv local 3.10.13
   $py = pyenv which python
   & $py -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

3. Install dependencies:

   ```powershell
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```

## Option B: Using winget

```powershell
winget install -e --id Python.Python.3.10
$py = (Get-Command py).Source
& $py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Verify TensorFlow/Keras

```powershell
.\.venv\Scripts\Activate.ps1
python .\temscript\check_tf_keras.py
```

You should see TensorFlow and Keras versions and a message: "Model built and compiled successfully."
