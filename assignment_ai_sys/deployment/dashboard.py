"""
📊 Model Performance Dashboard & Analytics
Comprehensive monitoring, user feedback, and performance analysis for DDI prediction models
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json
from datetime import datetime
import os

st.set_page_config(page_title="DDI Models Dashboard", layout="wide", page_icon="📊")

# Auto-refresh configuration
import time as time_module
AUTO_REFRESH_INTERVAL = 30  # seconds

# ============================================
# SESSION STATE INITIALIZATION
# ============================================
if 'prediction_history' not in st.session_state:
    st.session_state.prediction_history = []

if 'user_feedback' not in st.session_state:
    st.session_state.user_feedback = []

# ============================================
# DATA PERSISTENCE FUNCTIONS
# ============================================
HISTORY_FILE = "prediction_history.json"
FEEDBACK_FILE = "user_feedback.json"

def load_history():
    """Load prediction history from file"""
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, 'r') as f:
                return json.load(f)
        except:
            return []
    return []

def save_history(history):
    """Save prediction history to file"""
    with open(HISTORY_FILE, 'w') as f:
        json.dump(history, f, indent=2)

def load_feedback():
    """Load user feedback from file"""
    if os.path.exists(FEEDBACK_FILE):
        try:
            with open(FEEDBACK_FILE, 'r') as f:
                return json.load(f)
        except:
            return []
    return []

def save_feedback(feedback):
    """Save user feedback to file"""
    with open(FEEDBACK_FILE, 'w') as f:
        json.dump(feedback, f, indent=2)

# Load data on startup
if not st.session_state.prediction_history:
    st.session_state.prediction_history = load_history()

if not st.session_state.user_feedback:
    st.session_state.user_feedback = load_feedback()

# ============================================
# HEADER
# ============================================
st.title("📊 Drug-Drug Interaction Models Dashboard")
st.markdown("**Comprehensive Model Monitoring, Performance Analysis & User Feedback**")

# Real-time stats bar
col1, col2, col3 = st.columns([3, 1, 1])
with col1:
    st.markdown("**🔄 Auto-refreshing data from app12.py predictions**")
with col2:
    if st.button("🔄 Refresh Now"):
        st.session_state.prediction_history = load_history()
        st.session_state.user_feedback = load_feedback()
        st.rerun()
with col3:
    auto_refresh = st.checkbox("Auto-refresh (30s)", value=False)

if auto_refresh:
    time_module.sleep(AUTO_REFRESH_INTERVAL)
    st.session_state.prediction_history = load_history()
    st.session_state.user_feedback = load_feedback()
    st.rerun()

st.markdown("---")

# ============================================
# SIDEBAR - DATA COLLECTION
# ============================================
st.sidebar.header("📝 Add New Prediction Data")

with st.sidebar.form("add_prediction"):
    st.markdown("### Enter Prediction Details")

    drug1 = st.text_input("Drug 1 ID", placeholder="DB00001")
    drug2 = st.text_input("Drug 2 ID", placeholder="DB00002")

    model_used = st.selectbox("Model Used", ["DGNN", "Random Forest", "XGBoost"])

    predicted_type = st.number_input("Predicted Interaction Type", min_value=1, max_value=86, value=1)
    confidence = st.slider("Confidence (%)", 0.0, 100.0, 50.0, 0.01)
    prediction_time = st.number_input("Prediction Time (seconds)", min_value=0.001, value=0.1, format="%.3f")

    has_ground_truth = st.checkbox("I know the ground truth")
    ground_truth = None
    if has_ground_truth:
        ground_truth = st.number_input("Ground Truth Type", min_value=1, max_value=86, value=1)

    submitted = st.form_submit_button("➕ Add Prediction")

    if submitted and drug1 and drug2:
        prediction_data = {
            "timestamp": datetime.now().isoformat(),
            "drug1": drug1,
            "drug2": drug2,
            "model": model_used,
            "predicted_type": int(predicted_type),
            "confidence": float(confidence),
            "prediction_time": float(prediction_time),
            "ground_truth": int(ground_truth) if ground_truth else None,
            "correct": (int(predicted_type) == int(ground_truth)) if ground_truth else None
        }
        st.session_state.prediction_history.append(prediction_data)
        save_history(st.session_state.prediction_history)
        st.sidebar.success(f"✅ Added prediction for {drug1} + {drug2}")

st.sidebar.markdown("---")

# User Feedback Form
st.sidebar.header("⭐ Submit Feedback")

with st.sidebar.form("submit_feedback"):
    st.markdown("### Rate a Prediction")

    if st.session_state.prediction_history:
        recent_predictions = [
            f"{p['drug1']} + {p['drug2']} ({p['model']}) - Type {p['predicted_type']}"
            for p in st.session_state.prediction_history[-10:]
        ]
        selected_pred = st.selectbox("Select Prediction", recent_predictions)
    else:
        st.info("No predictions yet")
        selected_pred = None

    rating = st.slider("Rating (1-5 stars)", 1, 5, 3)
    accuracy_rating = st.select_slider(
        "How accurate was the prediction?",
        options=["Very Inaccurate", "Inaccurate", "Neutral", "Accurate", "Very Accurate"]
    )
    comments = st.text_area("Comments (optional)", placeholder="Share your experience...")

    submitted_feedback = st.form_submit_button("📨 Submit Feedback")

    if submitted_feedback and selected_pred:
        feedback_data = {
            "timestamp": datetime.now().isoformat(),
            "prediction": selected_pred,
            "rating": rating,
            "accuracy_rating": accuracy_rating,
            "comments": comments
        }
        st.session_state.user_feedback.append(feedback_data)
        save_feedback(st.session_state.user_feedback)
        st.sidebar.success("✅ Feedback submitted!")

# ============================================
# MAIN DASHBOARD TABS
# ============================================
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📈 Overview",
    "🔬 Model Comparison",
    "📊 Performance Metrics",
    "🎯 Ground Truth Analysis",
    "💬 User Feedback",
    "📜 Prediction History"
])

# ============================================
# TAB 1: OVERVIEW
# ============================================
with tab1:
    st.header("📈 Dashboard Overview")

    if not st.session_state.prediction_history:
        st.info("👋 Welcome! Start by adding prediction data using the sidebar form.")
        st.markdown("""
        ### How to use this dashboard:
        1. **Add Predictions**: Use the sidebar form to log predictions from any model
        2. **Include Ground Truth**: When available, add the actual interaction type
        3. **Submit Feedback**: Rate predictions and share your observations
        4. **Analyze Performance**: Explore the tabs to see detailed analytics
        """)
    else:
        df = pd.DataFrame(st.session_state.prediction_history)

        # Key Metrics Row
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Total Predictions", len(df))

        with col2:
            if 'ground_truth' in df.columns:
                with_gt = df['ground_truth'].notna().sum()
                st.metric("With Ground Truth", with_gt)
            else:
                st.metric("With Ground Truth", 0)

        with col3:
            avg_conf = df['confidence'].mean()
            st.metric("Avg Confidence", f"{avg_conf:.2f}%")

        with col4:
            if 'correct' in df.columns and df['correct'].notna().sum() > 0:
                accuracy = (df['correct'].sum() / df['correct'].notna().sum()) * 100
                st.metric("Overall Accuracy", f"{accuracy:.2f}%", delta=f"{accuracy-80:.1f}%")
            else:
                st.metric("Overall Accuracy", "N/A")

        st.markdown("---")

        # Charts Row
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📊 Predictions by Model")
            model_counts = df['model'].value_counts()

            fig = go.Figure(data=[
                go.Bar(
                    x=model_counts.index,
                    y=model_counts.values,
                    marker_color=['#667eea', '#f093fb', '#764ba2'],
                    text=model_counts.values,
                    textposition='auto'
                )
            ])
            fig.update_layout(
                xaxis_title="Model",
                yaxis_title="Number of Predictions",
                height=300,
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.subheader("⏱️ Average Prediction Time")
            avg_time_by_model = df.groupby('model')['prediction_time'].mean()

            fig = go.Figure(data=[
                go.Bar(
                    x=avg_time_by_model.index,
                    y=avg_time_by_model.values,
                    marker_color=['#667eea', '#f093fb', '#764ba2'],
                    text=[f"{t:.3f}s" for t in avg_time_by_model.values],
                    textposition='auto'
                )
            ])
            fig.update_layout(
                xaxis_title="Model",
                yaxis_title="Time (seconds)",
                height=300,
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)

        # Recent Predictions
        st.subheader("🕒 Recent Predictions")
        recent_df = df.tail(5)[['timestamp', 'drug1', 'drug2', 'model', 'predicted_type', 'confidence', 'ground_truth', 'correct']]
        recent_df['timestamp'] = pd.to_datetime(recent_df['timestamp']).dt.strftime('%Y-%m-%d %H:%M:%S')
        st.dataframe(recent_df, use_container_width=True)

# ============================================
# TAB 2: MODEL COMPARISON
# ============================================
with tab2:
    st.header("🔬 Model Comparison")

    if not st.session_state.prediction_history:
        st.info("No prediction data available yet.")
    else:
        df = pd.DataFrame(st.session_state.prediction_history)

        # Model Statistics Table
        st.subheader("📋 Model Statistics Summary")

        model_stats = []
        for model in df['model'].unique():
            model_df = df[df['model'] == model]

            stats = {
                "Model": model,
                "Total Predictions": len(model_df),
                "Avg Confidence": f"{model_df['confidence'].mean():.2f}%",
                "Avg Time (s)": f"{model_df['prediction_time'].mean():.3f}",
                "Min Time (s)": f"{model_df['prediction_time'].min():.3f}",
                "Max Time (s)": f"{model_df['prediction_time'].max():.3f}",
            }

            if 'correct' in model_df.columns and model_df['correct'].notna().sum() > 0:
                accuracy = (model_df['correct'].sum() / model_df['correct'].notna().sum()) * 100
                stats["Accuracy"] = f"{accuracy:.2f}%"
                stats["Correct"] = int(model_df['correct'].sum())
                stats["Total Tested"] = int(model_df['correct'].notna().sum())
            else:
                stats["Accuracy"] = "N/A"
                stats["Correct"] = 0
                stats["Total Tested"] = 0

            model_stats.append(stats)

        stats_df = pd.DataFrame(model_stats)
        st.dataframe(stats_df, use_container_width=True)

        st.markdown("---")

        # Side-by-side comparison charts
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("⚡ Speed Comparison")

            fig = go.Figure()

            for model in df['model'].unique():
                model_df = df[df['model'] == model]
                fig.add_trace(go.Box(
                    y=model_df['prediction_time'],
                    name=model,
                    boxmean='sd'
                ))

            fig.update_layout(
                yaxis_title="Prediction Time (seconds)",
                height=400,
                showlegend=True
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.subheader("🎯 Confidence Distribution")

            fig = go.Figure()

            for model in df['model'].unique():
                model_df = df[df['model'] == model]
                fig.add_trace(go.Box(
                    y=model_df['confidence'],
                    name=model,
                    boxmean='sd'
                ))

            fig.update_layout(
                yaxis_title="Confidence (%)",
                height=400,
                showlegend=True
            )
            st.plotly_chart(fig, use_container_width=True)

        # Accuracy comparison (if ground truth available)
        if 'correct' in df.columns and df['correct'].notna().sum() > 0:
            st.markdown("---")
            st.subheader("✅ Accuracy Comparison")

            accuracy_data = []
            for model in df['model'].unique():
                model_df = df[df['model'] == model]
                if model_df['correct'].notna().sum() > 0:
                    accuracy = (model_df['correct'].sum() / model_df['correct'].notna().sum()) * 100
                    accuracy_data.append({
                        "Model": model,
                        "Accuracy": accuracy
                    })

            if accuracy_data:
                acc_df = pd.DataFrame(accuracy_data)

                fig = go.Figure(data=[
                    go.Bar(
                        x=acc_df['Model'],
                        y=acc_df['Accuracy'],
                        text=[f"{a:.2f}%" for a in acc_df['Accuracy']],
                        textposition='auto',
                        marker_color=['#667eea', '#f093fb', '#764ba2']
                    )
                ])

                fig.update_layout(
                    xaxis_title="Model",
                    yaxis_title="Accuracy (%)",
                    height=400,
                    yaxis_range=[0, 100]
                )
                st.plotly_chart(fig, use_container_width=True)

# ============================================
# TAB 3: PERFORMANCE METRICS
# ============================================
with tab3:
    st.header("📊 Performance Metrics")

    if not st.session_state.prediction_history:
        st.info("No prediction data available yet.")
    else:
        df = pd.DataFrame(st.session_state.prediction_history)
        df['timestamp'] = pd.to_datetime(df['timestamp'])

        # Model selector
        selected_model = st.selectbox("Select Model to Analyze", ["All Models"] + list(df['model'].unique()))

        if selected_model != "All Models":
            df_filtered = df[df['model'] == selected_model]
        else:
            df_filtered = df

        st.markdown("---")

        # Time series analysis
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📈 Predictions Over Time")

            df_filtered['date'] = df_filtered['timestamp'].dt.date
            predictions_per_day = df_filtered.groupby('date').size().reset_index(name='count')

            fig = go.Figure(data=[
                go.Scatter(
                    x=predictions_per_day['date'],
                    y=predictions_per_day['count'],
                    mode='lines+markers',
                    line=dict(color='#667eea', width=3),
                    marker=dict(size=8)
                )
            ])

            fig.update_layout(
                xaxis_title="Date",
                yaxis_title="Number of Predictions",
                height=350
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.subheader("⏱️ Performance Trend")

            df_filtered_sorted = df_filtered.sort_values('timestamp')

            fig = go.Figure(data=[
                go.Scatter(
                    x=df_filtered_sorted.index,
                    y=df_filtered_sorted['prediction_time'],
                    mode='lines+markers',
                    line=dict(color='#f093fb', width=2),
                    marker=dict(size=6)
                )
            ])

            fig.update_layout(
                xaxis_title="Prediction Number",
                yaxis_title="Time (seconds)",
                height=350
            )
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")

        # Distribution analysis
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📊 Confidence Score Distribution")

            fig = go.Figure(data=[
                go.Histogram(
                    x=df_filtered['confidence'],
                    nbinsx=20,
                    marker_color='#667eea'
                )
            ])

            fig.update_layout(
                xaxis_title="Confidence (%)",
                yaxis_title="Frequency",
                height=350
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.subheader("🎯 Predicted Interaction Types")

            type_counts = df_filtered['predicted_type'].value_counts().head(10)

            fig = go.Figure(data=[
                go.Bar(
                    x=type_counts.index,
                    y=type_counts.values,
                    marker_color='#764ba2'
                )
            ])

            fig.update_layout(
                xaxis_title="Interaction Type",
                yaxis_title="Frequency",
                height=350
            )
            st.plotly_chart(fig, use_container_width=True)

# ============================================
# TAB 4: GROUND TRUTH ANALYSIS
# ============================================
with tab4:
    st.header("🎯 Ground Truth Analysis")

    if not st.session_state.prediction_history:
        st.info("No prediction data available yet.")
    else:
        df = pd.DataFrame(st.session_state.prediction_history)

        # Filter for entries with ground truth
        df_with_gt = df[df['ground_truth'].notna()]

        if len(df_with_gt) == 0:
            st.warning("⚠️ No predictions with ground truth available yet. Add ground truth values using the sidebar form.")
        else:
            st.success(f"✅ {len(df_with_gt)} predictions with ground truth available")

            # Overall accuracy metrics
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                total_correct = df_with_gt['correct'].sum()
                st.metric("Correct Predictions", int(total_correct))

            with col2:
                total_incorrect = len(df_with_gt) - total_correct
                st.metric("Incorrect Predictions", int(total_incorrect))

            with col3:
                overall_accuracy = (total_correct / len(df_with_gt)) * 100
                st.metric("Overall Accuracy", f"{overall_accuracy:.2f}%")

            with col4:
                avg_conf_correct = df_with_gt[df_with_gt['correct'] == True]['confidence'].mean()
                st.metric("Avg Confidence (Correct)", f"{avg_conf_correct:.2f}%")

            st.markdown("---")

            # Accuracy by model
            st.subheader("📊 Accuracy by Model")

            model_accuracy = []
            for model in df_with_gt['model'].unique():
                model_df = df_with_gt[df_with_gt['model'] == model]
                accuracy = (model_df['correct'].sum() / len(model_df)) * 100
                model_accuracy.append({
                    "Model": model,
                    "Accuracy": accuracy,
                    "Correct": int(model_df['correct'].sum()),
                    "Total": len(model_df)
                })

            acc_df = pd.DataFrame(model_accuracy)

            col1, col2 = st.columns([1, 2])

            with col1:
                st.dataframe(acc_df, use_container_width=True)

            with col2:
                fig = go.Figure(data=[
                    go.Bar(
                        x=acc_df['Model'],
                        y=acc_df['Accuracy'],
                        text=[f"{a:.1f}%" for a in acc_df['Accuracy']],
                        textposition='auto',
                        marker_color=['#667eea', '#f093fb', '#764ba2']
                    )
                ])

                fig.update_layout(
                    xaxis_title="Model",
                    yaxis_title="Accuracy (%)",
                    height=300,
                    yaxis_range=[0, 100]
                )
                st.plotly_chart(fig, use_container_width=True)

            st.markdown("---")

            # Confusion analysis
            st.subheader("🔍 Prediction Analysis")

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**Confidence vs Correctness**")

                fig = go.Figure()

                correct_df = df_with_gt[df_with_gt['correct'] == True]
                incorrect_df = df_with_gt[df_with_gt['correct'] == False]

                fig.add_trace(go.Box(
                    y=correct_df['confidence'],
                    name='Correct',
                    marker_color='green'
                ))

                fig.add_trace(go.Box(
                    y=incorrect_df['confidence'],
                    name='Incorrect',
                    marker_color='red'
                ))

                fig.update_layout(
                    yaxis_title="Confidence (%)",
                    height=350
                )
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                st.markdown("**Prediction Error Distribution**")

                df_with_gt['error'] = abs(df_with_gt['predicted_type'] - df_with_gt['ground_truth'])

                fig = go.Figure(data=[
                    go.Histogram(
                        x=df_with_gt['error'],
                        nbinsx=20,
                        marker_color='#f093fb'
                    )
                ])

                fig.update_layout(
                    xaxis_title="Prediction Error (|Predicted - Truth|)",
                    yaxis_title="Frequency",
                    height=350
                )
                st.plotly_chart(fig, use_container_width=True)

            st.markdown("---")

            # Detailed results table
            st.subheader("📋 Detailed Results")

            results_df = df_with_gt[['timestamp', 'drug1', 'drug2', 'model', 'predicted_type', 'ground_truth', 'confidence', 'correct']].copy()
            results_df['timestamp'] = pd.to_datetime(results_df['timestamp']).dt.strftime('%Y-%m-%d %H:%M:%S')
            results_df['correct'] = results_df['correct'].map({True: '✅', False: '❌'})

            st.dataframe(results_df, use_container_width=True)

# ============================================
# TAB 5: USER FEEDBACK
# ============================================
with tab5:
    st.header("💬 User Feedback Analysis")

    if not st.session_state.user_feedback:
        st.info("No user feedback submitted yet. Use the sidebar form to submit feedback.")
    else:
        feedback_df = pd.DataFrame(st.session_state.user_feedback)

        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Total Feedback", len(feedback_df))

        with col2:
            avg_rating = feedback_df['rating'].mean()
            st.metric("Average Rating", f"{avg_rating:.2f} ⭐")

        with col3:
            accuracy_map = {
                "Very Inaccurate": 1,
                "Inaccurate": 2,
                "Neutral": 3,
                "Accurate": 4,
                "Very Accurate": 5
            }
            feedback_df['accuracy_score'] = feedback_df['accuracy_rating'].map(accuracy_map)
            avg_accuracy = feedback_df['accuracy_score'].mean()
            st.metric("Avg Accuracy Rating", f"{avg_accuracy:.2f}/5")

        with col4:
            with_comments = feedback_df['comments'].notna().sum()
            st.metric("With Comments", with_comments)

        st.markdown("---")

        # Visualizations
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("⭐ Rating Distribution")

            rating_counts = feedback_df['rating'].value_counts().sort_index()

            fig = go.Figure(data=[
                go.Bar(
                    x=rating_counts.index,
                    y=rating_counts.values,
                    marker_color='#667eea',
                    text=rating_counts.values,
                    textposition='auto'
                )
            ])

            fig.update_layout(
                xaxis_title="Rating (Stars)",
                yaxis_title="Count",
                height=350
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.subheader("🎯 Accuracy Rating Distribution")

            accuracy_counts = feedback_df['accuracy_rating'].value_counts()

            fig = go.Figure(data=[
                go.Bar(
                    x=accuracy_counts.index,
                    y=accuracy_counts.values,
                    marker_color='#f093fb',
                    text=accuracy_counts.values,
                    textposition='auto'
                )
            ])

            fig.update_layout(
                xaxis_title="Accuracy Rating",
                yaxis_title="Count",
                height=350
            )
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")

        # Recent feedback
        st.subheader("📝 Recent Feedback")

        recent_feedback = feedback_df.tail(10).copy()
        recent_feedback['timestamp'] = pd.to_datetime(recent_feedback['timestamp']).dt.strftime('%Y-%m-%d %H:%M:%S')

        for idx, row in recent_feedback.iterrows():
            with st.expander(f"⭐ {row['rating']} stars - {row['timestamp']}"):
                st.markdown(f"**Prediction:** {row['prediction']}")
                st.markdown(f"**Accuracy Rating:** {row['accuracy_rating']}")
                if row['comments']:
                    st.markdown(f"**Comments:** {row['comments']}")

# ============================================
# TAB 6: PREDICTION HISTORY
# ============================================
with tab6:
    st.header("📜 Complete Prediction History")

    if not st.session_state.prediction_history:
        st.info("No prediction history available yet.")
    else:
        df = pd.DataFrame(st.session_state.prediction_history)
        df['timestamp'] = pd.to_datetime(df['timestamp']).dt.strftime('%Y-%m-%d %H:%M:%S')

        # Filters
        col1, col2, col3 = st.columns(3)

        with col1:
            model_filter = st.multiselect("Filter by Model", options=df['model'].unique(), default=df['model'].unique())

        with col2:
            min_conf = st.slider("Min Confidence (%)", 0.0, 100.0, 0.0)

        with col3:
            show_gt_only = st.checkbox("Show only with Ground Truth")

        # Apply filters
        df_filtered = df[df['model'].isin(model_filter)]
        df_filtered = df_filtered[df_filtered['confidence'] >= min_conf]

        if show_gt_only:
            df_filtered = df_filtered[df_filtered['ground_truth'].notna()]

        st.markdown(f"**Showing {len(df_filtered)} of {len(df)} predictions**")

        # Display table
        st.dataframe(df_filtered, use_container_width=True)

        # Export button
        csv = df_filtered.to_csv(index=False)
        st.download_button(
            label="📥 Download as CSV",
            data=csv,
            file_name=f"prediction_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )

# ============================================
# FOOTER
# ============================================
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>📊 DDI Models Dashboard | Built with Streamlit | Real-time Model Monitoring & Analytics</p>
</div>
""", unsafe_allow_html=True)
