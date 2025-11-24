# Drug-Drug Interaction Prediction System (DGNN-DDI)

[![Python 3.13](https://img.shields.io/badge/python-3.13-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/streamlit-1.28+-red.svg)](https://streamlit.io/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

A comprehensive machine learning system for predicting drug-drug interactions using three state-of-the-art models: Directed Graph Neural Network (DGNN), Random Forest, and XGBoost. The system features real-time prediction, interactive 3D molecular visualization, and a comprehensive analytics dashboard.

## 🌟 Features

### Main Application (app12.py)
- **Three ML Models**: DGNN (95.39% accuracy), Random Forest (96.25% accuracy), XGBoost (95.65% accuracy)
- **Multiple Prediction Modes**:
  - Single drug pair prediction
  - Batch prediction from CSV
  - Interaction matrix generation
  - Drug screening against dataset
- **Interactive 3D Molecular Visualization** using py3Dmol
- **2D Molecular Structure Display** using RDKit
- **Automatic Prediction Logging** for analytics
- **Detailed Model Information** including feature processing and performance metrics

### Analytics Dashboard (dashboard.py)
- **Real-time Monitoring**: Auto-refresh every 30 seconds
- **6 Comprehensive Tabs**:
  1. **Overview**: Total predictions, model distribution, recent activity
  2. **Model Comparison**: Side-by-side performance analysis
  3. **Performance Metrics**: Trends over time, confidence distributions
  4. **Ground Truth Analysis**: Accuracy tracking with actual labels
  5. **User Feedback**: Rating system and comments
  6. **Prediction History**: Complete log with filtering and export
- **Interactive Plotly Charts**: Bar charts, histograms, scatter plots, time series
- **CSV Export Functionality** for all data

## 📊 System Performance

| Model | Test Accuracy | Macro Precision | Macro Recall | Macro F1 | Model Size | Speed |
|-------|---------------|-----------------|--------------|----------|------------|-------|
| **DGNN** | 95.39% | 94.50% | 93.80% | 94.15% | 7.4 MB | 0.2-0.5s |
| **Random Forest** | 96.25% | 98.77% | 93.18% | 95.26% | 8.4 GB | 0.1-0.3s |
| **XGBoost** | 95.65% | 97.68% | 95.59% | 96.25% | 113 MB | 0.1-0.3s |

## Models

### 1. DGNN (Directed Graph Neural Network) - Primary Model
- **Configuration:**
  - PyTorch-based graph neural network
  - Message passing on molecular graphs
  - Test Accuracy: **95.39%**
  - Validation Accuracy: **95.48%**

### 2. Random Forest Classifier
- **Configuration:**
  - `n_estimators=200`
  - `max_depth=None`
  - `random_state=42`
  - `n_jobs=-1`
  - Test Accuracy: **96.25%**

### 3. XGBoost Classifier
- **Configuration:**
  - `n_estimators=500`
  - `max_depth=8`
  - `learning_rate=0.05`
  - `subsample=0.8`
  - `colsample_bytree=0.8`
  - `random_state=42`
  - Test Accuracy: **95.65%**

## Features Used for Training

Both models were trained on the following 11 features:

1. **MolWt_1** - Molecular weight of drug 1
2. **MolWt_2** - Molecular weight of drug 2
3. **LogP_1** - Lipophilicity (partition coefficient) of drug 1
4. **LogP_2** - Lipophilicity of drug 2
5. **HBD_1** - Hydrogen bond donors in drug 1
6. **HBD_2** - Hydrogen bond donors in drug 2
7. **HBA_1** - Hydrogen bond acceptors in drug 1
8. **HBA_2** - Hydrogen bond acceptors in drug 2
9. **TPSA_1** - Topological polar surface area of drug 1
10. **TPSA_2** - Topological polar surface area of drug 2
11. **Fingerprint_Similarity** - Tanimoto similarity between Morgan fingerprints (radius=2, nBits=1024)

### Feature Extraction Process

```
Drug ID → SMILES → RDKit Mol Object → Molecular Descriptors + Fingerprints
```

## Prediction Pipeline

### Step 1: Input
- User enters two drug IDs (e.g., DB00001, DB00002)

### Step 2: SMILES Lookup
- App retrieves SMILES strings from drug database (drugbank100.csv)
- Database contains 1,706 unique drugs with 191,808 drug pair interactions

### Step 3: Feature Extraction
```python
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors, AllChem, DataStructs

# Parse SMILES
mol1 = Chem.MolFromSmiles(smiles1)
mol2 = Chem.MolFromSmiles(smiles2)

# Calculate descriptors
MolWt = Descriptors.MolWt(mol)
LogP = Descriptors.MolLogP(mol)
HBD = rdMolDescriptors.CalcNumHBD(mol)
HBA = rdMolDescriptors.CalcNumHBA(mol)
TPSA = rdMolDescriptors.CalcTPSA(mol)

# Calculate fingerprint similarity
fp1 = AllChem.GetMorganFingerprintAsBitVect(mol1, 2, nBits=1024)
fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2, 2, nBits=1024)
similarity = DataStructs.TanimotoSimilarity(fp1, fp2)
```

### Step 4: Prediction
- Features are passed to the XGBoost model
- Model outputs:
  - Predicted interaction type (class label)
  - Probability distribution across all classes
  - Confidence score (max probability)

### Step 5: Output
- Displays molecular structures (2D)
- Shows feature comparison table
- Presents prediction results with confidence scores
- Lists top 5 most likely interaction types

## Files

- `app.py` - Main Streamlit application
- `predict_ddi.py` - Command-line prediction script
- `drugbank100.csv` - Drug database with SMILES strings
- `ddi_xgb_model.pkl` - Trained XGBoost model (113MB) ✅
- `ddi_rf_model.pkl` - Trained Random Forest model (14GB) ⚠️ Corrupted
- `dxr.ipynb` - Training notebook

## How to Run

### Streamlit App
```bash
cd /Users/sathyadharinisrinivasan/Desktop/xrb
streamlit run app.py
```

### Command Line Script
```bash
cd /Users/sathyadharinisrinivasan/Desktop/xrb
python predict_ddi.py
```

## Database Structure

**drugbank100.csv columns:**
- `ID1` - First drug ID
- `ID2` - Second drug ID
- `Y` - Interaction type (class label)
- `X1` - SMILES string for drug 1
- `X2` - SMILES string for drug 2
- `Map` - Interaction description
- `Map1` - Additional mapping info

## Model Performance Notes

### Training Setup
- **Train/Test Split:** 80/20
- **Random State:** 42
- **Stratified Sampling:** Yes (maintains class distribution)

### Current Status
- **XGBoost Model:**
- **Random Forest Model:**
- **GNN Model**

## Fixing the Random Forest Model

To retrain a working RF model:

1. Load the training data
2. Train with same parameters but consider:
   - Reducing `n_estimators` (try 100-200 instead of 300)
   - Setting `max_depth` to limit tree size
   - Using `max_samples` to reduce memory usage

Example:
```python
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=20,
    max_samples=0.8,
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train, y_train)
```

## Dependencies

```
streamlit
pandas
rdkit
scikit-learn
xgboost
pickle
gnn
randomforest
tesorflow
```


