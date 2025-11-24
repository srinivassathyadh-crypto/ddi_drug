import streamlit as st
import pandas as pd
import numpy as np
import torch
import os
from drug_predictor import DrugInteractionPredictor
import sys
sys.path.insert(0, '.')
from data_preprocessing import CustomData
from rdkit import Chem
from rdkit.Chem import Draw, Descriptors, AllChem, rdMolDescriptors, DataStructs
import matplotlib.pyplot as plt
import py3Dmol
from stmol import showmol
import streamlit.components.v1 as components
import pickle
import warnings
warnings.filterwarnings('ignore')
from prediction_logger import log_prediction

st.set_page_config(page_title="DGNN-DDI Predictor", layout="wide")

# Helper functions for visualization
def draw_molecule_3d(smiles):
    """Draw interactive 3D molecular structure using py3Dmol"""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        # Generate 3D coordinates
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, AllChem.ETKDG())
        AllChem.UFFOptimizeMolecule(mol)

        # Convert to MOL block
        mol_block = Chem.MolToMolBlock(mol)

        # Create py3Dmol view
        view = py3Dmol.view(width=600, height=500)
        view.addModel(mol_block, 'mol')
        view.setStyle({'stick': {'radius': 0.18}, 'sphere': {'radius': 0.27}})
        view.setBackgroundColor('white')
        view.zoomTo()
        view.spin(True)  # Enable auto-rotation

        return view
    except Exception as e:
        return None

st.title("🔬 Drug-Drug Interaction Prediction (DGNN-DDI)")
st.write("Predict interactions using your trained Graph Neural Network model.")

# ---------------------------
# Sidebar Config
# ---------------------------
st.sidebar.header("Configuration")

model_path = st.sidebar.text_input(
    "Model Weights (.pt)",
    "epoch20_valacc0.9548_testacc0.9539.pt.pt"
)

drug_data_path = st.sidebar.text_input(
    "Drug Data (.pkl)",
    "drug_data.pkl"
)

device = st.sidebar.selectbox("Device", ["cpu", "cuda" if torch.cuda.is_available() else "cpu"])

# Load predictor safely
@st.cache_resource(show_spinner=True)
def load_predictor(model_path, drug_data_path, device):
    return DrugInteractionPredictor(model_path, drug_data_path, device)

if os.path.exists(model_path) and os.path.exists(drug_data_path):
    predictor = load_predictor(model_path, drug_data_path, device)
else:
    st.error("❌ Model or drug_data path invalid.")
    st.stop()


# -----------------------------------------
#            Mode Selection
# -----------------------------------------
mode = st.selectbox(
    "Select Mode",
    ["Single Prediction", "Batch Prediction", "Matrix Generation", "Screen Drug", "RF/XGBoost Models", "📊 Dashboard & Analytics"]
)

# -----------------------------------------
#            Single Prediction
# -----------------------------------------
if mode == "Single Prediction":
    st.subheader("🔍 Single Drug Pair Prediction")

    drug1 = st.text_input("Drug 1 ID (e.g., DB00460)")
    drug2 = st.text_input("Drug 2 ID (e.g., DB04571)")

    col1, col2 = st.columns(2)
    with col1:
        detailed_view = st.checkbox("Show Detailed Analysis", value=True)
    with col2:
        show_all_interactions = st.checkbox("Show All 86 Interactions", value=False)

    # Model selection buttons
    st.markdown("### 🎯 Select Prediction Model:")
    col1, col2, col3 = st.columns(3)

    with col1:
        dgnn_button = st.button("🔬 DGNN Prediction", type="primary", use_container_width=True)
    with col2:
        rf_button = st.button("🌲 Random Forest", use_container_width=True)
    with col3:
        xgb_button = st.button("⚡ XGBoost", use_container_width=True)

    if dgnn_button:
        import time
        start_time = time.time()

        with st.spinner("Running prediction…"):
            try:
                if detailed_view:
                    # Get detailed prediction with SMILES and features
                    result = predictor.predict_pair_detailed(drug1, drug2)

                    prediction_time = time.time() - start_time

                    # Display Drug Information with Molecular Structures
                    st.markdown("---")
                    st.markdown("### 💊 Drug Information & Molecular Structure")

                    col1, col2 = st.columns(2)

                    with col1:
                        st.markdown(f"**Drug 1: {result['drug1_id']}**")
                        st.markdown("**SMILES:**")
                        st.code(result['drug1_smiles'], language=None)

                        # 3D Structure
                        st.markdown("**3D Molecular Structure:**")
                        mol_view1 = draw_molecule_3d(result['drug1_smiles'])
                        if mol_view1:
                            showmol(mol_view1, height=500, width=600)
                        else:
                            st.warning("3D structure could not be generated")

                        st.markdown("**Molecular Graph Features:**")
                        st.json(result['drug1_features'])

                    with col2:
                        st.markdown(f"**Drug 2: {result['drug2_id']}**")
                        st.markdown("**SMILES:**")
                        st.code(result['drug2_smiles'], language=None)

                        # 3D Structure
                        st.markdown("**3D Molecular Structure:**")
                        mol_view2 = draw_molecule_3d(result['drug2_smiles'])
                        if mol_view2:
                            showmol(mol_view2, height=500, width=600)
                        else:
                            st.warning("3D structure could not be generated")

                        st.markdown("**Molecular Graph Features:**")
                        st.json(result['drug2_features'])

                    # Model Details and Feature Processing
                    st.markdown("---")
                    st.markdown("### 🔬 DGNN Model Details & Feature Processing")

                    with st.expander("📊 View Model Architecture & Processing Steps", expanded=True):
                        st.markdown("""
                        **Model Type:** D-MPNN (Directed Message Passing Neural Network)

                        **Training Details:**
                        - Dataset: DrugBank drug-drug interactions
                        - Classes: 86 interaction types
                        - Train/Test Split: 80/20
                        - Random State: 42
                        """)

                        st.markdown("**🔄 Feature Extraction Pipeline:**")
                        st.markdown("""
                        1. **SMILES → RDKit Molecule Object**
                           - Parse chemical structure from SMILES string
                           - Validate molecular structure

                        2. **Molecular Graph Construction**
                           - **Nodes (Atoms):**
                             - Atomic number, degree, formal charge
                             - Hybridization type, aromaticity
                             - Total H count, chirality
                           - **Edges (Bonds):**
                             - Bond type (single, double, triple, aromatic)
                             - Conjugation, ring membership
                             - Stereochemistry

                        3. **Line Graph Transformation**
                           - Bonds become nodes in line graph
                           - Bond-bond relationships encoded as edges
                           - Enables message passing along bond paths

                        4. **D-MPNN Processing**
                           - **Message Passing (3 iterations):**
                             - Hidden dimension: 64
                             - Directed message aggregation
                             - Edge-to-edge information flow
                           - **Co-Attention Mechanism:**
                             - Cross-drug attention weights
                             - Captures interaction patterns
                           - **Global Pooling:**
                             - Aggregates bond/node features
                             - Creates fixed-size drug representation

                        5. **RESCAL Scoring**
                           - Relation-specific scoring for 86 interaction types
                           - Learns interaction-specific patterns
                           - Sigmoid activation for probabilities
                        """)

                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Drug 1 Atoms", result['drug1_features']['num_atoms'])
                            st.metric("Drug 1 Bonds", result['drug1_features']['num_bonds'])
                        with col2:
                            st.metric("Drug 2 Atoms", result['drug2_features']['num_atoms'])
                            st.metric("Drug 2 Bonds", result['drug2_features']['num_bonds'])
                        with col3:
                            st.metric("Prediction Time", f"{prediction_time:.3f}s")
                            st.metric("Model Accuracy", "95.39%")

                        st.markdown(f"""
                        **Feature Dimensions:**
                        - Node features per atom: {result['drug1_features']['node_feature_dim']}D
                        - Edge features per bond: {result['drug1_features']['edge_feature_dim']}D
                        - Total features processed: {(result['drug1_features']['num_atoms'] + result['drug2_features']['num_atoms']) * result['drug1_features']['node_feature_dim'] + (result['drug1_features']['num_bonds'] + result['drug2_features']['num_bonds']) * result['drug1_features']['edge_feature_dim']}
                        """)

                    # Display Top Predicted Interaction (Highlighted)
                    st.markdown("---")
                    st.markdown("### 🎯 Predicted Interaction Result")

                    top = result['top_interaction']

                    # Create a highlighted box for the top interaction
                    st.markdown(
                        f"""
                        <div style="
                            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                            padding: 25px;
                            border-radius: 15px;
                            color: white;
                            box-shadow: 0 10px 25px rgba(0,0,0,0.2);
                            margin: 20px 0;
                        ">
                            <h2 style="margin: 0 0 15px 0; font-size: 24px;">
                                🏆 Most Likely Interaction Type
                            </h2>
                            <div style="font-size: 48px; font-weight: bold; margin: 15px 0;">
                                Type {top['relation_type']}
                            </div>
                            <div style="font-size: 20px; margin: 10px 0; line-height: 1.4;">
                                {top['relation_name']}
                            </div>
                            <div style="margin-top: 20px; padding: 15px; background: rgba(255,255,255,0.2); border-radius: 10px;">
                                <div style="font-size: 18px; margin: 5px 0;">
                                    <strong>Probability:</strong> {top['probability']:.4f} ({top['probability']*100:.2f}%)
                                </div>
                                <div style="font-size: 18px; margin: 5px 0;">
                                    <strong>Prediction:</strong> {top['prediction']}
                                </div>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                    # Progress bar for probability
                    st.markdown("**Interaction Probability:**")
                    st.progress(float(top['probability']))

                    # Log prediction automatically
                    log_prediction(
                        drug1=drug1,
                        drug2=drug2,
                        model="DGNN",
                        predicted_type=top['relation_type'],
                        confidence=top['probability'] * 100,
                        prediction_time=prediction_time
                    )

                    # Clinical Testing Section for High Probability Interactions
                    high_prob_interactions = [i for i in result['all_interactions'] if i['probability'] > 0.9]

                    if high_prob_interactions:
                        st.markdown("---")
                        st.markdown("### 🚨 Clinical Testing Required")

                        st.markdown("""
                        <div style="background: linear-gradient(135deg, #e74c3c 0%, #c0392b 100%); padding: 25px; border-radius: 15px; color: white; border: 3px solid #c0392b; box-shadow: 0 5px 15px rgba(231, 76, 60, 0.3);">
                            <h3 style="margin: 0 0 15px 0;">⚠️ HIGH PROBABILITY INTERACTIONS DETECTED (>90%)</h3>
                            <p style="margin: 0; font-size: 16px;">The following interactions show very high probability and require clinical testing and validation:</p>
                        </div>
                        """, unsafe_allow_html=True)

                        # Create a dataframe for clinical testing
                        df_clinical = pd.DataFrame(high_prob_interactions)
                        df_clinical['probability_pct'] = df_clinical['probability'].apply(lambda x: f"{x*100:.2f}%")
                        df_clinical['clinical_status'] = '🔴 REQUIRES TESTING'

                        st.markdown("""
                        <style>
                        .clinical-table {
                            background-color: #ffe6e6;
                            border: 2px solid #e74c3c;
                            border-radius: 10px;
                            padding: 15px;
                        }
                        </style>
                        """, unsafe_allow_html=True)

                        st.dataframe(
                            df_clinical[['relation_type', 'relation_name', 'probability_pct', 'clinical_status']],
                            use_container_width=True,
                            height=min(len(high_prob_interactions) * 50 + 100, 400)
                        )

                        st.warning(f"""
                        **⚠️ Clinical Recommendation:**
                        - **{len(high_prob_interactions)} interaction type(s)** show probability > 90%
                        - These predictions require experimental validation
                        - Recommended: In vitro studies → Clinical trials
                        - Do NOT use for medical decisions without proper validation
                        """)

                    # Display All Interactions
                    if show_all_interactions:
                        st.markdown("---")
                        st.markdown("### 📊 All 86 Interaction Types (Sorted by Probability)")

                        df_all = pd.DataFrame(result['all_interactions'])

                        # Format probability as percentage
                        df_all['probability_pct'] = df_all['probability'].apply(lambda x: f"{x*100:.2f}%")

                        # Display with conditional formatting
                        st.dataframe(
                            df_all[['relation_type', 'relation_name', 'probability_pct', 'prediction']],
                            use_container_width=True,
                            height=400
                        )

                        # Show top 10 as bar chart
                        st.markdown("**Top 10 Most Likely Interactions:**")
                        top_10 = df_all.head(10)
                        st.bar_chart(
                            top_10.set_index('relation_type')['probability'],
                            use_container_width=True
                        )
                    else:
                        # Show just top 5
                        st.markdown("---")
                        st.markdown("### 📈 Top 5 Most Likely Interactions")
                        df_top5 = pd.DataFrame(result['all_interactions'][:5])
                        df_top5['probability_pct'] = df_top5['probability'].apply(lambda x: f"{x*100:.2f}%")
                        st.table(df_top5[['relation_type', 'relation_name', 'probability_pct', 'prediction']])

                else:
                    # Simple mode
                    rel_type = st.number_input("Relation Type (1–86)", 1, 86, 1)
                    result = predictor.predict_pair(drug1, drug2, rel_type)
                    st.json(result)

            except Exception as e:
                st.error(f"❌ Error: {str(e)}")

    # -----------------------------------------
    #  Random Forest & XGBoost Predictions
    # -----------------------------------------

    # Load RF/XGBoost models and database (only once)
    @st.cache_resource
    def load_rf_xgb_models():
        """Load RF and XGBoost models"""
        models = {}
        xrb_path = "/Users/sathyadharinisrinivasan/Desktop/xrb"

        try:
            with open(f"{xrb_path}/ddi_rf_model.pkl", "rb") as f:
                models['rf'] = pickle.load(f)
        except Exception as e:
            models['rf'] = None

        try:
            with open(f"{xrb_path}/ddi_xgb_model.pkl", "rb") as f:
                models['xgb'] = pickle.load(f)
        except Exception as e:
            models['xgb'] = None

        return models

    @st.cache_data
    def load_drug_database_rfxgb():
        """Load drug database for RF/XGBoost"""
        xrb_path = "/Users/sathyadharinisrinivasan/Desktop/xrb"
        try:
            df = pd.read_csv(f"{xrb_path}/drugbank100.csv")
            drug_db = {}

            for _, row in df.iterrows():
                if pd.notna(row['ID1']) and pd.notna(row['X1']):
                    drug_db[str(row['ID1'])] = row['X1']
                if pd.notna(row['ID2']) and pd.notna(row['X2']):
                    drug_db[str(row['ID2'])] = row['X2']

            return drug_db, df
        except Exception as e:
            return {}, None

    def extract_features_rfxgb(smiles1, smiles2):
        """Extract molecular features from SMILES"""
        mol1 = Chem.MolFromSmiles(smiles1)
        mol2 = Chem.MolFromSmiles(smiles2)

        if mol1 is None or mol2 is None:
            raise ValueError("Invalid SMILES")

        features = {
            'MolWt_1': Descriptors.MolWt(mol1),
            'MolWt_2': Descriptors.MolWt(mol2),
            'LogP_1': Descriptors.MolLogP(mol1),
            'LogP_2': Descriptors.MolLogP(mol2),
            'HBD_1': rdMolDescriptors.CalcNumHBD(mol1),
            'HBD_2': rdMolDescriptors.CalcNumHBD(mol2),
            'HBA_1': rdMolDescriptors.CalcNumHBA(mol1),
            'HBA_2': rdMolDescriptors.CalcNumHBA(mol2),
            'TPSA_1': rdMolDescriptors.CalcTPSA(mol1),
            'TPSA_2': rdMolDescriptors.CalcTPSA(mol2),
        }

        fp1 = AllChem.GetMorganFingerprintAsBitVect(mol1, 2, nBits=1024)
        fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2, 2, nBits=1024)
        features['Fingerprint_Similarity'] = DataStructs.TanimotoSimilarity(fp1, fp2)

        return pd.DataFrame([features]), mol1, mol2

    # Handle RF button click
    if rf_button:
        rf_xgb_models = load_rf_xgb_models()
        drug_db_rfxgb, _ = load_drug_database_rfxgb()

        if not drug_db_rfxgb:
            st.error("❌ Drug database not available for RF/XGBoost")
        elif rf_xgb_models['rf'] is None:
            st.error("❌ Random Forest model not loaded")
        elif drug1 not in drug_db_rfxgb or drug2 not in drug_db_rfxgb:
            st.error("❌ One or both drugs not found in RF/XGBoost database")
        else:
            with st.spinner("Running Random Forest prediction..."):
                try:
                    import time
                    start_time = time.time()

                    smiles1 = drug_db_rfxgb[drug1]
                    smiles2 = drug_db_rfxgb[drug2]
                    features, mol1, mol2 = extract_features_rfxgb(smiles1, smiles2)

                    # Prediction
                    rf_pred = rf_xgb_models['rf'].predict(features)[0]
                    rf_proba = rf_xgb_models['rf'].predict_proba(features)[0]
                    rf_conf = max(rf_proba)

                    prediction_time = time.time() - start_time

                    # Display results
                    st.markdown("---")
                    st.markdown("### 🌲 Random Forest Prediction Results")

                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Predicted Interaction Type", int(rf_pred))
                    with col2:
                        st.metric("Confidence", f"{rf_conf:.4f}", delta=f"{rf_conf*100:.2f}%")

                    st.progress(float(rf_conf))

                    # Top 10 interactions
                    st.markdown("### 📈 Top 10 Most Likely Interactions")
                    rf_top = np.argsort(rf_proba)[-10:][::-1]
                    rf_top_df = pd.DataFrame({
                        'Interaction Type': rf_top,
                        'Probability': [rf_proba[i] for i in rf_top],
                        'Percentage': [f"{rf_proba[i]*100:.2f}%" for i in rf_top]
                    })
                    st.dataframe(rf_top_df, use_container_width=True)

                    st.markdown("**Probability Distribution:**")
                    st.bar_chart(
                        rf_top_df.set_index('Interaction Type')['Probability'],
                        use_container_width=True,
                        color='#667eea'
                    )

                    # Log prediction automatically
                    log_prediction(
                        drug1=drug1,
                        drug2=drug2,
                        model="Random Forest",
                        predicted_type=int(rf_pred),
                        confidence=rf_conf * 100,
                        prediction_time=prediction_time
                    )

                    # Model Details and Feature Processing
                    st.markdown("---")
                    st.markdown("### 🌲 Random Forest Model Details & Feature Processing")

                    with st.expander("📊 View Model Architecture & Processing Steps", expanded=True):
                        st.markdown("""
                        **Model Type:** Random Forest Classifier

                        **Training Details:**
                        - Dataset: DrugBank drug-drug interactions
                        - Total Samples: 191,808 drug pairs
                        - Classes: 86 interaction types (Y values: 1-86)
                        - Train/Test Split: 80/20
                        - Random State: 42

                        **Model Hyperparameters:**
                        - Number of Estimators: 200 trees
                        - Max Depth: 30
                        - Max Samples: 0.8 (bootstrap sampling)
                        - n_jobs: -1 (parallel processing)
                        """)

                        st.markdown("---")
                        st.markdown("### 🔬 Feature Extraction Pipeline (SMILES → Features)")

                        st.markdown("""
                        **Step 1: SMILES Input**
                        - Drug 1 SMILES: `{}`
                        - Drug 2 SMILES: `{}`

                        **Step 2: RDKit Molecule Object Creation**
                        - Convert SMILES strings to RDKit molecule objects
                        - Validate molecular structure

                        **Step 3: Molecular Descriptor Calculation**

                        For each drug, calculate 5 molecular descriptors:
                        """.format(smiles1[:50] + "..." if len(smiles1) > 50 else smiles1,
                                  smiles2[:50] + "..." if len(smiles2) > 50 else smiles2))

                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown("""
                            **Drug 1 Descriptors:**
                            - **MolWt_1**: Molecular Weight = `{:.2f}` Da
                            - **LogP_1**: Partition Coefficient = `{:.2f}`
                            - **HBD_1**: H-Bond Donors = `{}`
                            - **HBA_1**: H-Bond Acceptors = `{}`
                            - **TPSA_1**: Topological Polar Surface Area = `{:.2f}` ų
                            """.format(features['MolWt_1'].values[0],
                                      features['LogP_1'].values[0],
                                      int(features['HBD_1'].values[0]),
                                      int(features['HBA_1'].values[0]),
                                      features['TPSA_1'].values[0]))

                        with col2:
                            st.markdown("""
                            **Drug 2 Descriptors:**
                            - **MolWt_2**: Molecular Weight = `{:.2f}` Da
                            - **LogP_2**: Partition Coefficient = `{:.2f}`
                            - **HBD_2**: H-Bond Donors = `{}`
                            - **HBA_2**: H-Bond Acceptors = `{}`
                            - **TPSA_2**: Topological Polar Surface Area = `{:.2f}` ų
                            """.format(features['MolWt_2'].values[0],
                                      features['LogP_2'].values[0],
                                      int(features['HBD_2'].values[0]),
                                      int(features['HBA_2'].values[0]),
                                      features['TPSA_2'].values[0]))

                        st.markdown("""
                        **Step 4: Fingerprint Generation & Similarity**
                        - Generate Morgan Fingerprints (Circular FPs):
                          - Radius: 2
                          - nBits: 1024
                        - Calculate Tanimoto Similarity between fingerprints
                        - **Fingerprint_Similarity**: `{:.4f}`

                        **Step 5: Feature Vector Assembly**
                        - Total Features: 11 (10 molecular descriptors + 1 similarity score)
                        - Feature Vector Shape: (1, 11)
                        - All features normalized by the model internally

                        **Step 6: Ensemble Prediction**
                        - Each of 200 trees votes on the interaction type
                        - Aggregate votes to determine final prediction
                        - Calculate probability distribution across 86 classes
                        """.format(features['Fingerprint_Similarity'].values[0]))

                        st.markdown("---")
                        st.markdown("### 📈 Performance Metrics")

                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Prediction Time", f"{prediction_time:.3f}s")
                        with col2:
                            st.metric("Model Accuracy", "82.26%")
                        with col3:
                            st.metric("Training Accuracy", "99.64%")

                        st.markdown("""
                        **Note:** Random Forest uses Y values directly (1-86 range).
                        No transformation is applied to predictions.
                        """)

                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")

    # Handle XGBoost button click
    if xgb_button:
        rf_xgb_models = load_rf_xgb_models()
        drug_db_rfxgb, _ = load_drug_database_rfxgb()

        if not drug_db_rfxgb:
            st.error("❌ Drug database not available for RF/XGBoost")
        elif rf_xgb_models['xgb'] is None:
            st.error("❌ XGBoost model not loaded")
        elif drug1 not in drug_db_rfxgb or drug2 not in drug_db_rfxgb:
            st.error("❌ One or both drugs not found in RF/XGBoost database")
        else:
            with st.spinner("Running XGBoost prediction..."):
                try:
                    import time
                    start_time = time.time()

                    smiles1 = drug_db_rfxgb[drug1]
                    smiles2 = drug_db_rfxgb[drug2]
                    features, mol1, mol2 = extract_features_rfxgb(smiles1, smiles2)

                    # Prediction
                    xgb_pred = rf_xgb_models['xgb'].predict(features)[0]
                    xgb_proba = rf_xgb_models['xgb'].predict_proba(features)[0]
                    xgb_conf = max(xgb_proba)
                    actual_pred = int(xgb_pred) + 1  # Add 1 for XGBoost

                    prediction_time = time.time() - start_time

                    # Display results
                    st.markdown("---")
                    st.markdown("### ⚡ XGBoost Prediction Results")

                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Predicted Interaction Type", actual_pred)
                    with col2:
                        st.metric("Confidence", f"{xgb_conf:.4f}", delta=f"{xgb_conf*100:.2f}%")

                    st.progress(float(xgb_conf))

                    # Top 10 interactions
                    st.markdown("### 📈 Top 10 Most Likely Interactions")
                    xgb_top = np.argsort(xgb_proba)[-10:][::-1]
                    xgb_top_df = pd.DataFrame({
                        'Interaction Type': [i+1 for i in xgb_top],
                        'Probability': [xgb_proba[i] for i in xgb_top],
                        'Percentage': [f"{xgb_proba[i]*100:.2f}%" for i in xgb_top]
                    })
                    st.dataframe(xgb_top_df, use_container_width=True)

                    st.markdown("**Probability Distribution:**")
                    st.bar_chart(
                        xgb_top_df.set_index('Interaction Type')['Probability'],
                        use_container_width=True,
                        color='#764ba2'
                    )

                    # Log prediction automatically
                    log_prediction(
                        drug1=drug1,
                        drug2=drug2,
                        model="XGBoost",
                        predicted_type=actual_pred,
                        confidence=xgb_conf * 100,
                        prediction_time=prediction_time
                    )

                    # Model Details and Feature Processing
                    st.markdown("---")
                    st.markdown("### ⚡ XGBoost Model Details & Feature Processing")

                    with st.expander("📊 View Model Architecture & Processing Steps", expanded=True):
                        st.markdown("""
                        **Model Type:** XGBoost Classifier (Gradient Boosting)

                        **Training Details:**
                        - Dataset: DrugBank drug-drug interactions
                        - Total Samples: 191,808 drug pairs
                        - Classes: 86 interaction types
                        - **Target Encoding:** Y-1 (shifted to 0-85 range)
                        - Train/Test Split: 80/20
                        - Random State: 42

                        **Model Hyperparameters:**
                        - Number of Estimators: 500 boosting rounds
                        - Max Depth: 8
                        - Learning Rate: 0.05
                        - Subsample: 0.8
                        - Colsample by Tree: 0.8
                        - n_jobs: -1 (parallel processing)
                        """)

                        st.markdown("---")
                        st.markdown("### 🔬 Feature Extraction Pipeline (SMILES → Features)")

                        st.markdown("""
                        **Step 1: SMILES Input**
                        - Drug 1 SMILES: `{}`
                        - Drug 2 SMILES: `{}`

                        **Step 2: RDKit Molecule Object Creation**
                        - Convert SMILES strings to RDKit molecule objects
                        - Validate molecular structure

                        **Step 3: Molecular Descriptor Calculation**

                        For each drug, calculate 5 molecular descriptors:
                        """.format(smiles1[:50] + "..." if len(smiles1) > 50 else smiles1,
                                  smiles2[:50] + "..." if len(smiles2) > 50 else smiles2))

                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown("""
                            **Drug 1 Descriptors:**
                            - **MolWt_1**: Molecular Weight = `{:.2f}` Da
                            - **LogP_1**: Partition Coefficient = `{:.2f}`
                            - **HBD_1**: H-Bond Donors = `{}`
                            - **HBA_1**: H-Bond Acceptors = `{}`
                            - **TPSA_1**: Topological Polar Surface Area = `{:.2f}` ų
                            """.format(features['MolWt_1'].values[0],
                                      features['LogP_1'].values[0],
                                      int(features['HBD_1'].values[0]),
                                      int(features['HBA_1'].values[0]),
                                      features['TPSA_1'].values[0]))

                        with col2:
                            st.markdown("""
                            **Drug 2 Descriptors:**
                            - **MolWt_2**: Molecular Weight = `{:.2f}` Da
                            - **LogP_2**: Partition Coefficient = `{:.2f}`
                            - **HBD_2**: H-Bond Donors = `{}`
                            - **HBA_2**: H-Bond Acceptors = `{}`
                            - **TPSA_2**: Topological Polar Surface Area = `{:.2f}` ų
                            """.format(features['MolWt_2'].values[0],
                                      features['LogP_2'].values[0],
                                      int(features['HBD_2'].values[0]),
                                      int(features['HBA_2'].values[0]),
                                      features['TPSA_2'].values[0]))

                        st.markdown("""
                        **Step 4: Fingerprint Generation & Similarity**
                        - Generate Morgan Fingerprints (Circular FPs):
                          - Radius: 2
                          - nBits: 1024
                        - Calculate Tanimoto Similarity between fingerprints
                        - **Fingerprint_Similarity**: `{:.4f}`

                        **Step 5: Feature Vector Assembly**
                        - Total Features: 11 (10 molecular descriptors + 1 similarity score)
                        - Feature Vector Shape: (1, 11)
                        - All features normalized by the model internally

                        **Step 6: Gradient Boosting Prediction**
                        - Sequential tree building (500 rounds)
                        - Each tree corrects errors from previous trees
                        - Learning rate: 0.05 controls the contribution of each tree
                        - Final prediction: Weighted sum of all tree predictions
                        - Raw Prediction Range: 0-85 (requires +1 adjustment)
                        - **Final Prediction**: {} (raw: {} + 1)
                        """.format(features['Fingerprint_Similarity'].values[0],
                                  actual_pred, int(xgb_pred)))

                        st.markdown("---")
                        st.markdown("### 📈 Performance Metrics")

                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Prediction Time", f"{prediction_time:.3f}s")
                        with col2:
                            st.metric("Model Size", "113 MB")
                        with col3:
                            st.metric("Classes", "86 (0-85)")

                        st.markdown("""
                        **Important Note:** XGBoost was trained with Y-1 encoding (0-85 range).
                        Predictions are automatically adjusted by adding +1 to convert back to the original Y values (1-86).
                        """)

                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")


# -----------------------------------------
#             Batch Prediction
# -----------------------------------------
elif mode == "Batch Prediction":
    st.subheader("📂 Batch Prediction from CSV (ID1, ID2, Y?)")

    file = st.file_uploader("Upload CSV", type=["csv"])

    if file:
        df = pd.read_csv(file)
        st.write("Input Preview:")
        st.dataframe(df.head())

        if st.button("Run Batch Prediction"):
            pairs = list(zip(df["ID1"], df["ID2"]))
            relations = df["Y"].tolist() if "Y" in df.columns else None

            with st.spinner("Predicting…"):
                results = predictor.predict_batch(pairs, relations)
                results_df = pd.DataFrame(results)

            st.write("Results:")
            st.dataframe(results_df)

            st.download_button(
                "Download Output CSV",
                results_df.to_csv(index=False).encode("utf-8"),
                "batch_predictions.csv"
            )


# -----------------------------------------
#           Interaction Matrix
# -----------------------------------------
elif mode == "Matrix Generation":
    st.subheader("🧪 Generate Interaction Probability Matrix")

    top_n = st.number_input("Top N Drugs", 5, 200, 20)
    relation_type = st.number_input("Relation Type", 1, 86, 1)

    if st.button("Generate Matrix"):
        with st.spinner("Computing matrix… may take time"):
            matrix = predictor.generate_interaction_matrix(
                top_n=top_n,
                relation_type=relation_type
            )

        st.write("Matrix:")
        st.dataframe(matrix)

        st.download_button(
            "Download Matrix CSV",
            matrix.to_csv().encode("utf-8"),
            "interaction_matrix.csv"
        )


# -----------------------------------------
#             Screen a Drug
# -----------------------------------------
elif mode == "Screen Drug":
    st.subheader("🔎 Screen All Interactions for a Target Drug")

    drug_id = st.text_input("Target Drug ID")
    threshold = st.slider("Threshold", 0.0, 1.0, 0.5)
    relation_type = st.number_input("Relation Type", 1, 86, 1)
    top_k = st.number_input("Top K (optional)", 1, 200, 20)

    if st.button("Find Interactions"):
        with st.spinner("Screening…"):
            df = predictor.find_interactions_for_drug(
                drug_id,
                threshold=threshold,
                relation_type=relation_type,
                top_k=top_k
            )

        st.write("Interactions Found:")
        st.dataframe(df)

        st.download_button(
            "Download CSV",
            df.to_csv(index=False).encode("utf-8"),
            "screen_results.csv"
        )

# -----------------------------------------
#    RF/XGBoost Models Mode
# -----------------------------------------
elif mode == "RF/XGBoost Models":
    st.subheader("🌲⚡ Random Forest & XGBoost Prediction")
    st.write("Predict interactions using Random Forest and XGBoost models trained on molecular descriptors")
    
    # Load RF/XGBoost models
    @st.cache_resource
    def load_rf_xgb_models():
        """Load RF and XGBoost models"""
        models = {}
        xrb_path = "/Users/sathyadharinisrinivasan/Desktop/xrb"
        
        try:
            with open(f"{xrb_path}/ddi_rf_model.pkl", "rb") as f:
                models['rf'] = pickle.load(f)
            st.sidebar.success("✅ Random Forest model loaded")
        except Exception as e:
            st.sidebar.warning(f"⚠️ RF model not available")
            models['rf'] = None
        
        try:
            with open(f"{xrb_path}/ddi_xgb_model.pkl", "rb") as f:
                models['xgb'] = pickle.load(f)
            st.sidebar.success("✅ XGBoost model loaded")
        except Exception as e:
            st.sidebar.error(f"⚠️ XGBoost error: {e}")
            models['xgb'] = None
        
        return models
    
    @st.cache_data
    def load_drug_database_rfxgb():
        """Load drug database for RF/XGBoost"""
        xrb_path = "/Users/sathyadharinisrinivasan/Desktop/xrb"
        try:
            df = pd.read_csv(f"{xrb_path}/drugbank100.csv")
            drug_db = {}
            
            for _, row in df.iterrows():
                if pd.notna(row['ID1']) and pd.notna(row['X1']):
                    drug_db[str(row['ID1'])] = row['X1']
                if pd.notna(row['ID2']) and pd.notna(row['X2']):
                    drug_db[str(row['ID2'])] = row['X2']
            
            st.sidebar.success(f"✅ Loaded {len(drug_db)} unique drugs")
            return drug_db, df
        except Exception as e:
            st.sidebar.error(f"⚠️ Could not load database: {e}")
            return {}, None
    
    def extract_features_rfxgb(smiles1, smiles2):
        """Extract molecular features from SMILES"""
        mol1 = Chem.MolFromSmiles(smiles1)
        mol2 = Chem.MolFromSmiles(smiles2)
        
        if mol1 is None or mol2 is None:
            raise ValueError("Invalid SMILES")
        
        features = {
            'MolWt_1': Descriptors.MolWt(mol1),
            'MolWt_2': Descriptors.MolWt(mol2),
            'LogP_1': Descriptors.MolLogP(mol1),
            'LogP_2': Descriptors.MolLogP(mol2),
            'HBD_1': rdMolDescriptors.CalcNumHBD(mol1),
            'HBD_2': rdMolDescriptors.CalcNumHBD(mol2),
            'HBA_1': rdMolDescriptors.CalcNumHBA(mol1),
            'HBA_2': rdMolDescriptors.CalcNumHBA(mol2),
            'TPSA_1': rdMolDescriptors.CalcTPSA(mol1),
            'TPSA_2': rdMolDescriptors.CalcTPSA(mol2),
        }
        
        fp1 = AllChem.GetMorganFingerprintAsBitVect(mol1, 2, nBits=1024)
        fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2, 2, nBits=1024)
        features['Fingerprint_Similarity'] = DataStructs.TanimotoSimilarity(fp1, fp2)
        
        return pd.DataFrame([features]), mol1, mol2
    
    # Load models and database
    rf_xgb_models = load_rf_xgb_models()
    drug_db_rfxgb, full_df_rfxgb = load_drug_database_rfxgb()
    
    if not drug_db_rfxgb:
        st.error("❌ No drug database available")
        st.stop()
    
    if rf_xgb_models['rf'] is None and rf_xgb_models['xgb'] is None:
        st.error("❌ No models loaded")
        st.stop()
    
    # Input section
    st.markdown("### 🔍 Enter Drug Pair")
    
    col1, col2 = st.columns(2)
    with col1:
        drug1_rf = st.selectbox("Select Drug 1", sorted(drug_db_rfxgb.keys()), key="rf_drug1")
    with col2:
        drug2_rf = st.selectbox("Select Drug 2", sorted(drug_db_rfxgb.keys()), key="rf_drug2")
    
    col1, col2 = st.columns(2)
    with col1:
        detailed_view_rf = st.checkbox("Show Detailed Analysis", value=True, key="rf_detailed")
    with col2:
        show_all_interactions_rf = st.checkbox("Show All Interaction Probabilities", value=False, key="rf_showall")
    
    if st.button("🔬 Predict Interaction", type="primary", use_container_width=True, key="rf_predict"):
        if not drug1_rf or not drug2_rf:
            st.warning("⚠️ Please select both drugs")
        elif drug1_rf not in drug_db_rfxgb or drug2_rf not in drug_db_rfxgb:
            st.error("❌ Drug not found in database")
        else:
            with st.spinner("Running prediction..."):
                try:
                    # Get SMILES
                    smiles1 = drug_db_rfxgb[drug1_rf]
                    smiles2 = drug_db_rfxgb[drug2_rf]
                    
                    # Extract features
                    features, mol1, mol2 = extract_features_rfxgb(smiles1, smiles2)
                    
                    # Display drug information
                    if detailed_view_rf:
                        st.markdown("---")
                        st.markdown("### 💊 Drug Information")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown(f"**Drug 1: {drug1_rf}**")
                            st.code(smiles1, language=None)
                            
                            # 3D structure
                            view = draw_molecule_3d(smiles1)
                            if view:
                                showmol(view, height=500, width=600)
                        
                        with col2:
                            st.markdown(f"**Drug 2: {drug2_rf}**")
                            st.code(smiles2, language=None)
                            
                            view = draw_molecule_3d(smiles2)
                            if view:
                                showmol(view, height=500, width=600)
                        
                        # Features table
                        st.markdown("---")
                        st.markdown("### 📊 Molecular Features")
                        
                        feature_data = []
                        for col in features.columns:
                            if '_1' in col:
                                feature_name = col.replace('_1', '')
                                drug1_val = features[col].values[0]
                                drug2_col = col.replace('_1', '_2')
                                drug2_val = features[drug2_col].values[0] if drug2_col in features.columns else None
                                
                                feature_data.append({
                                    'Feature': feature_name,
                                    f'{drug1_rf}': f"{drug1_val:.4f}",
                                    f'{drug2_rf}': f"{drug2_val:.4f}" if drug2_val is not None else "N/A"
                                })
                        
                        fp_sim = features['Fingerprint_Similarity'].values[0]
                        feature_data.append({
                            'Feature': 'Fingerprint Similarity',
                            f'{drug1_rf}': '',
                            f'{drug2_rf}': f"{fp_sim:.4f}"
                        })
                        
                        st.dataframe(pd.DataFrame(feature_data), use_container_width=True)
                        
                        st.metric("Molecular Similarity (Tanimoto)", f"{fp_sim:.4f}", delta=f"{fp_sim*100:.2f}%")
                    
                    # Predictions
                    st.markdown("---")
                    st.markdown("### 🎯 Prediction Results")
                    
                    col1, col2 = st.columns(2)
                    
                    results_rf = {}
                    
                    # Random Forest
                    if rf_xgb_models['rf'] is not None:
                        with col1:
                            st.markdown("**🌲 Random Forest**")
                            
                            rf_pred = rf_xgb_models['rf'].predict(features)[0]
                            rf_proba = rf_xgb_models['rf'].predict_proba(features)[0]
                            rf_conf = max(rf_proba)
                            
                            results_rf['rf'] = {'pred': int(rf_pred), 'conf': rf_conf}
                            
                            st.metric("Predicted Interaction Type", int(rf_pred))
                            st.metric("Confidence", f"{rf_conf:.4f}", delta=f"{rf_conf*100:.2f}%")
                            st.progress(float(rf_conf))
                    
                    # XGBoost
                    if rf_xgb_models['xgb'] is not None:
                        with col2:
                            st.markdown("**⚡ XGBoost**")
                            
                            xgb_pred = rf_xgb_models['xgb'].predict(features)[0]
                            xgb_proba = rf_xgb_models['xgb'].predict_proba(features)[0]
                            xgb_conf = max(xgb_proba)
                            actual_pred = int(xgb_pred) + 1
                            
                            results_rf['xgb'] = {'pred': actual_pred, 'conf': xgb_conf}
                            
                            st.metric("Predicted Interaction Type", actual_pred)
                            st.metric("Confidence", f"{xgb_conf:.4f}", delta=f"{xgb_conf*100:.2f}%")
                            st.progress(float(xgb_conf))
                    
                    # Clinical Testing Required section
                    high_prob_interactions = []
                    
                    if rf_xgb_models['rf'] is not None:
                        for i, prob in enumerate(rf_proba):
                            if prob > 0.9:
                                high_prob_interactions.append({
                                    'Model': 'Random Forest',
                                    'Interaction Type': i,
                                    'Probability': prob,
                                    'Percentage': f"{prob*100:.2f}%",
                                    'Status': '🔴 REQUIRES TESTING'
                                })
                    
                    if rf_xgb_models['xgb'] is not None:
                        for i, prob in enumerate(xgb_proba):
                            if prob > 0.9:
                                high_prob_interactions.append({
                                    'Model': 'XGBoost',
                                    'Interaction Type': i + 1,
                                    'Probability': prob,
                                    'Percentage': f"{prob*100:.2f}%",
                                    'Status': '🔴 REQUIRES TESTING'
                                })
                    
                    if high_prob_interactions:
                        st.markdown("---")
                        st.markdown("### 🚨 Clinical Testing Required")
                        
                        st.markdown("""
                        <div style="background: linear-gradient(135deg, #e74c3c 0%, #c0392b 100%); padding: 25px; border-radius: 15px; color: white; border: 3px solid #c0392b; box-shadow: 0 5px 15px rgba(231, 76, 60, 0.3);">
                            <h3 style="margin: 0 0 15px 0;">⚠️ HIGH PROBABILITY INTERACTIONS DETECTED (>90%)</h3>
                            <p style="margin: 0; font-size: 16px;">The following interactions show very high probability and require clinical testing and validation:</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        df_clinical = pd.DataFrame(high_prob_interactions)
                        
                        st.dataframe(
                            df_clinical[['Model', 'Interaction Type', 'Percentage', 'Status']],
                            use_container_width=True,
                            height=min(len(high_prob_interactions) * 50 + 100, 400)
                        )
                        
                        st.warning(f"""
                        **⚠️ Clinical Recommendation:**
                        - **{len(high_prob_interactions)} interaction type(s)** show probability > 90%
                        - These predictions require experimental validation
                        - Recommended: In vitro studies → Clinical trials
                        - Do NOT use for medical decisions without proper validation
                        """)
                    
                    # Model Agreement
                    st.markdown("---")
                    if rf_xgb_models['rf'] is not None and rf_xgb_models['xgb'] is not None:
                        if results_rf['rf']['pred'] == results_rf['xgb']['pred']:
                            st.success(f"✅ Both models agree on interaction type: **{results_rf['rf']['pred']}**")
                        else:
                            st.warning(f"⚠️ Models disagree: RF predicts **{results_rf['rf']['pred']}**, XGBoost predicts **{results_rf['xgb']['pred']}**")
                    
                    # Show top probabilities
                    if show_all_interactions_rf:
                        st.markdown("---")
                        st.markdown("### 📊 All Interaction Type Probabilities")
                        
                        if rf_xgb_models['rf'] is not None:
                            st.markdown("**Random Forest - All 86 Interaction Types:**")
                            rf_all_sorted = np.argsort(rf_proba)[::-1]
                            rf_all_df = pd.DataFrame({
                                'Interaction Type': rf_all_sorted,
                                'Probability': [rf_proba[i] for i in rf_all_sorted],
                                'Percentage': [f"{rf_proba[i]*100:.2f}%" for i in rf_all_sorted]
                            })
                            st.dataframe(rf_all_df, use_container_width=True, height=400)
                        
                        if rf_xgb_models['xgb'] is not None:
                            st.markdown("**XGBoost - All 86 Interaction Types:**")
                            xgb_all_sorted = np.argsort(xgb_proba)[::-1]
                            xgb_all_df = pd.DataFrame({
                                'Interaction Type': [i+1 for i in xgb_all_sorted],
                                'Probability': [xgb_proba[i] for i in xgb_all_sorted],
                                'Percentage': [f"{xgb_proba[i]*100:.2f}%" for i in xgb_all_sorted]
                            })
                            st.dataframe(xgb_all_df, use_container_width=True, height=400)
                    else:
                        # Show just top 10 with bar chart
                        st.markdown("---")
                        st.markdown("### 📈 Top 10 Most Likely Interactions")
                        
                        col1, col2 = st.columns(2)
                        
                        if rf_xgb_models['rf'] is not None:
                            with col1:
                                st.markdown("**Random Forest:**")
                                rf_top = np.argsort(rf_proba)[-10:][::-1]
                                rf_top_df = pd.DataFrame({
                                    'Interaction Type': rf_top,
                                    'Probability': [rf_proba[i] for i in rf_top],
                                    'Percentage': [f"{rf_proba[i]*100:.2f}%" for i in rf_top]
                                })
                                st.dataframe(rf_top_df, use_container_width=True)
                                
                                # Bar chart
                                st.markdown("**Probability Distribution:**")
                                st.bar_chart(
                                    rf_top_df.set_index('Interaction Type')['Probability'],
                                    use_container_width=True,
                                    color='#667eea'
                                )
                        
                        if rf_xgb_models['xgb'] is not None:
                            with col2:
                                st.markdown("**XGBoost:**")
                                xgb_top = np.argsort(xgb_proba)[-10:][::-1]
                                xgb_top_df = pd.DataFrame({
                                    'Interaction Type': [i+1 for i in xgb_top],
                                    'Probability': [xgb_proba[i] for i in xgb_top],
                                    'Percentage': [f"{xgb_proba[i]*100:.2f}%" for i in xgb_top]
                                })
                                st.dataframe(xgb_top_df, use_container_width=True)
                                
                                # Bar chart
                                st.markdown("**Probability Distribution:**")
                                st.bar_chart(
                                    xgb_top_df.set_index('Interaction Type')['Probability'],
                                    use_container_width=True,
                                    color='#764ba2'
                                )
                    
                except Exception as e:
                    st.error(f"❌ Error during prediction: {str(e)}")
                    import traceback
                    with st.expander("View Error Details"):
                        st.code(traceback.format_exc())
