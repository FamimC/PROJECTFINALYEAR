# Pneumonia Detection AI Project

## 🔧 Project Setup Instructions

> ⚠️ Note: To keep this repository lightweight, we have excluded the `.venv/Lib/site-packages` directory. Please follow the steps below to set up your own environment.

### 1. Create and Activate Virtual Environment
```bash
python -m venv .venv
# On Windows:
.venv\Scripts\activate
```

### 2. Install Required Libraries
Once the virtual environment is activated, install dependencies using:
```bash
pip install -r requirements.txt
```

> If `requirements.txt` is not available, please install the core packages manually:
```bash
pip install tensorflow opencv-python pydicom matplotlib numpy pandas kaggle kagglehub
```

### 3. Dataset Download (Kaggle)

You can download the dataset in two ways:

#### Option A: Use `kagglehub` (preferred for automated setup)
```python
import kagglehub

# Download the latest dataset version
path = kagglehub.dataset_download("paultimothymooney/chest-xray-pneumonia")

print("Path to dataset files:", path)
```

#### Option B: Manual Download  
Visit the dataset page and download it manually:  
🔗 https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia

Extract it to your desired path and update the dataset path in your code accordingly.

### 4. Environment Setup (Anaconda + Jupyter)

If using Anaconda, you can alternatively create a conda environment:
```bash
conda create --name pneumonia_env python=3.10
conda activate pneumonia_env
pip install -r requirements.txt
```

To run the project in a Jupyter Notebook:
```bash
pip install notebook
jupyter notebook
```

---

## 🗂 Directory Structure

```
project/
│
├── .venv/                <- Virtual environment (excluded from GitHub)
├── notebooks/            <- Jupyter notebooks
├── model/                <- Trained model files (.h5 etc.)
├── scripts/              <- Python scripts
├── dataset/              <- Dataset folder (excluded from GitHub)
├── requirements.txt
└── README.md
```

## ⚠️ Excluded Files from GitHub

- `.venv/` – Virtual environment (use `python -m venv` to recreate)
- `dataset/` – Dataset files (download via Kaggle)
- `__pycache__/`, `.ipynb_checkpoints/` – Auto-generated folders
- `*.h5` – Model files (optional to exclude if too large)

> Add these to your `.gitignore` file:
```
.venv/
dataset/
__pycache__/
*.h5
.ipynb_checkpoints/
```

---

## 💡 Notes
- This project was initially developed in Jupyter Notebooks inside an Anaconda environment.
- Ensure you have your `kaggle.json` API token in place if you use automated downloads.
- For GPU training, ensure your system is configured with CUDA-compatible TensorFlow and proper NVIDIA drivers.

---

## 📩 Contact  
For any queries or setup help, feel free to reach out!
