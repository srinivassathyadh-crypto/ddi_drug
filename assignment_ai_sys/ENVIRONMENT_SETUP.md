# 🔧 Environment Setup Guide

## Current Situation

You have successfully trained your model (which means you have the right environment somewhere!), but the current Python environment doesn't have the required packages.

---

## ⚠️ Important Note

Since you successfully ran `train.py` and trained the model for 20 epochs, you **already have** a working environment set up. You just need to activate it!

---

## 🔍 Finding Your Training Environment

You likely used one of these methods to train the model:

### Option 1: Conda Environment (Most Common)

Check if you have conda environments:
```bash
conda env list
```

Look for an environment name like:
- `dgnn`
- `pytorch`
- `drug`
- `torch`
- Or any custom name you created

**Activate it:**
```bash
conda activate <environment_name>
```

Then run the examples:
```bash
python quick_start_examples.py
```

### Option 2: Virtual Environment (venv)

Check if there's a virtual environment in the project:
```bash
ls -la | grep -E "venv|env|.env"
```

**Activate it:**
```bash
source venv/bin/activate  # or env/bin/activate
python quick_start_examples.py
```

### Option 3: System Python with Packages

If you installed packages globally, try:
```bash
pip3 list | grep torch
```

If you see torch listed, use `python3` instead:
```bash
python3 quick_start_examples.py
```

---

## 🛠️ Creating a New Environment (If Needed)

If you can't find your original environment, create a new one:

### Using Conda (Recommended)

```bash
# Create environment
conda create -n dgnn python=3.8 -y

# Activate it
conda activate dgnn

# Install packages
pip install torch==1.12.0
pip install torch-geometric==2.0.4
pip install torch-scatter==2.0.9 -f https://data.pyg.org/whl/torch-1.12.0+cpu.html
pip install torch-sparse==0.6.14 -f https://data.pyg.org/whl/torch-1.12.0+cpu.html
pip install pandas==1.4.3
pip install numpy==1.22.3
pip install scikit-learn==1.1.1
pip install tqdm==4.64.0
pip install rdkit-pypi==2022.3.5
```

### Using venv

```bash
# Create environment
python3 -m venv dgnn_env

# Activate it
source dgnn_env/bin/activate

# Install packages (same as above)
pip install torch==1.12.0
pip install torch-geometric==2.0.4
# ... etc
```

---

## 📦 Required Packages

Based on your `README.md`, you need:

```
numpy==1.22.3
pandas==1.4.3
python==3.8.13
pytorch==1.12.0
rdkit==2020.09.1
scikit-learn==1.1.1
torch-geometric==2.0.4
torch-scatter==2.0.9
torch-sparse==0.6.14
tqdm==4.64.0
```

---

## ✅ Quick Test

After activating your environment, test it:

```bash
python -c "import torch; import torch_geometric; print('Success! PyTorch version:', torch.__version__)"
```

Expected output:
```
Success! PyTorch version: 1.12.0
```

---

## 🚀 Once Environment is Ready

Run the examples:

```bash
# Test your setup
python quick_start_examples.py

# Or make a prediction
python predict.py --mode single --drug1 DB04571 --drug2 DB00460
```

---

## 🔍 Troubleshooting

### Problem: "conda: command not found"
**Solution**: You might be using venv or system Python. Try `python3` or look for a `venv` folder.

### Problem: "Can't find my environment"
**Solution**: Check how you originally ran `train.py`. Use the same method:
```bash
# Look at your shell history
history | grep train.py
```

### Problem: "Package versions conflict"
**Solution**: Use the exact versions from README.md or create a fresh environment.

### Problem: "Still can't find it"
**Solution**: Create a new environment using the instructions above. It will work the same way.

---

## 💡 Recommended Approach

Since you successfully trained the model, the **fastest solution** is:

1. **Check your recent commands**:
   ```bash
   history | grep -E "conda activate|source|python train.py"
   ```

2. **Look for common conda environments**:
   ```bash
   conda env list
   ```

3. **Try activating likely names**:
   ```bash
   conda activate dgnn
   # or
   conda activate pytorch
   # or
   conda activate base
   ```

4. **Test if it works**:
   ```bash
   python -c "import torch; print('Found it!')"
   ```

---

## 📞 Quick Help

**Found your environment?**
→ Activate it and run `python quick_start_examples.py`

**Can't find it?**
→ Create a new one using the "Creating a New Environment" section above

**Need to verify it works?**
→ Run: `python -c "import torch; print(torch.__version__)"`

---

## ✅ Next Steps After Setup

Once your environment is activated and working:

1. Run the examples:
   ```bash
   python quick_start_examples.py
   ```

2. Make your first prediction:
   ```bash
   python predict.py --mode single --drug1 DB04571 --drug2 DB00460
   ```

3. Continue with the guides in GETTING_STARTED.md

---

*Remember: You already have everything installed somewhere - you just need to activate that environment!*
