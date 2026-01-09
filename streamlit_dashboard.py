import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from database import connect_to_db, fetch_data, close_connection
import json
from sklearn.metrics import confusion_matrix, roc_curve, auc
import pandas as pd
import numpy as np


# Page configuration
st.set_page_config(
    page_title="Network Intrusion Detection Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    </style>
""", unsafe_allow_html=True)


# Load raw data samples
def load_raw_data(connection, split, limit=100):
    query = """
    SELECT * FROM raw_network_traffic
    WHERE dataset_split = %s
    ORDER BY id
    LIMIT %s
    """
    return fetch_data(connection, query, (split, limit))


# Load training runs from database
def load_training_runs(connection):
    query="SELECT * FROM training_runs ORDER BY created_at DESC"
    return fetch_data(connection, query)


# Load metrics for a given run_id
def load_metrics(connection, run_id):
    query="SELECT * FROM model_metrics WHERE run_id = %s"
    return fetch_data(connection, query, (run_id,))


# Load predictions for a given run_id
def load_predictions(connection, run_id):
    query="SELECT * FROM predictions WHERE run_id = %s ORDER BY record_id"
    return fetch_data(connection, query, (run_id,))



# Plot metrics comparison
def plot_metrics_comparison(runs_df, metrics_df):
    # Merge metrics with model names
    merged = metrics_df.merge(
        runs_df[['run_id', 'model_name']],
        on='run_id',
        how='left'
    )

    # Average metrics per model
    avg_df = (
        merged
        .groupby(['model_name', 'metric_name'])['metric_value']
        .mean()
        .reset_index()
    )

    # Pivot for plotting
    pivot_df = avg_df.pivot(
        index='model_name',
        columns='metric_name',
        values='metric_value'
    ).reset_index()

    metric_cols = [c for c in pivot_df.columns if c != 'model_name']

    fig = go.Figure()
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

    for idx, metric in enumerate(metric_cols):
        fig.add_trace(go.Bar(
            name=metric.replace('_', ' ').title(),
            x=pivot_df['model_name'],
            y=pivot_df[metric],
            text=pivot_df[metric].round(4),
            textposition='outside',
            marker_color=colors[idx % len(colors)],
            hovertemplate='<b>%{x}</b><br>%{y:.4f}<extra></extra>'
        ))

    fig.update_layout(
        title="Average Model Performance Comparison",
        xaxis_title="Model",
        yaxis_title="Average Score",
        barmode='group',
        height=500,
        template='plotly_white',
        legend=dict(
            orientation="h",
            y=1.1,
            x=1,
            xanchor="right"
        )
    )

    return fig


# Plot confusion matrix
def plot_confusion_matrix(y_true, y_pred, model_name):
    cm = confusion_matrix(y_true, y_pred)
    
    # Calculate percentages
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    labels = ['Normal', 'Attack']
    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=labels,
        y=labels,
        colorscale='Blues',
        text=[[f'{cm[i][j]}<br>({cm_normalized[i][j]:.1%})' 
               for j in range(len(labels))] 
              for i in range(len(labels))],
        texttemplate='%{text}',
        textfont={"size": 14},
        hovertemplate='Actual: %{y}<br>Predicted: %{x}<br>Count: %{z}<extra></extra>',
        showscale=True
    ))
    fig.update_layout(
        title=f'{model_name}',
        xaxis_title='Predicted Label',
        yaxis_title='Actual Label',
        height=400,
        template='plotly_white'
    )
    return fig


# Plot ROC curve
def plot_roc_curve(connection, run_ids, runs_df):
    fig = go.Figure()
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    for idx, run_id in enumerate(run_ids):
        # Load predictions
        preds_df = load_predictions(connection, run_id)
        
        # Load actual labels
        query = """
        SELECT r.id, r.label 
        FROM raw_network_traffic r
        WHERE r.dataset_split = 'test'
        ORDER BY r.id
        """
        actual_df = fetch_data(connection, query)
        
        # Merge predictions with actual labels
        merged = preds_df.merge(
            actual_df, 
            left_on='record_id', 
            right_on='id',
            how='inner'
        )
        
        if len(merged) == 0:
            continue
            
        y_true = merged['label']
        y_proba = merged['predicted_probability']
        
        # Calculate ROC curve
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        roc_auc = auc(fpr, tpr)
        
        model_name = runs_df[runs_df['run_id'] == run_id]['model_name'].values[0]
        
        fig.add_trace(go.Scatter(
            x=fpr,
            y=tpr,
            mode='lines',
            name=f'{model_name} (AUC = {roc_auc:.4f})',
            line=dict(color=colors[idx % len(colors)], width=2),
            hovertemplate='FPR: %{x:.4f}<br>TPR: %{y:.4f}<extra></extra>'
        ))
    
    # Add diagonal reference line
    fig.add_trace(go.Scatter(
        x=[0, 1],
        y=[0, 1],
        mode='lines',
        name='Random Classifier',
        line=dict(color='gray', width=2, dash='dash'),
        showlegend=True
    ))
    
    fig.update_layout(
        title={
            'text': 'ROC Curves',
            'font': {'size': 20, 'color': '#1f77b4'}
        },
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate',
        height=500,
        template='plotly_white',
        legend=dict(
            orientation="v",
            yanchor="bottom",
            y=0.05,
            xanchor="right",
            x=0.95
        ),
        hovermode='closest'
    )
    
    return fig


# Plot prediction distribution
def plot_prediction_distribution(preds_df, model_name):
    fig = go.Figure()
    
    fig.add_trace(go.Histogram(
        x=preds_df['predicted_probability'],
        nbinsx=50,
        name='Probability Distribution',
        marker_color='#1f77b4',
        hovertemplate='Probability: %{x:.3f}<br>Count: %{y}<extra></extra>'
    ))
    
    fig.update_layout(
        title=f'{model_name}',
        xaxis_title='Predicted Probability',
        yaxis_title='Count',
        height=400,
        template='plotly_white',
        showlegend=False
    )
    
    return fig


# Plot class distribution
def plot_class_distribution(df, title):
    class_counts=df['label'].value_counts()
    
    fig = go.Figure(data=[go.Pie(
        labels=['Normal', 'Attack'],
        values=[class_counts.get(0, 0), class_counts.get(1, 0)],
        hole=0.4,
        marker_colors=['#2ca02c', '#d62728'],
        textinfo='label+percent+value',
        hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>'
    )])
    
    fig.update_layout(
        title={
            'text': title,
            'font': {'size': 18}
        },
        height=400,
        template='plotly_white'
    )
    
    return fig


# Plot feature correlation heatmap for sample data
def plot_feature_correlation(df, sample_size=1000):
    # Select numeric columns only
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    numeric_cols = [col for col in numeric_cols if col not in ['id', 'label']]
    
    # Sample data if too large
    if len(df) > sample_size:
        df_sample = df[numeric_cols].sample(sample_size, random_state=42)
    else:
        df_sample = df[numeric_cols]
    
    # Calculate correlation matrix
    corr_matrix = df_sample.corr()
    
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        colorscale='RdBu',
        zmid=0,
        text=corr_matrix.values,
        hovertemplate='%{x} vs %{y}<br>Correlation: %{z:.2f}<extra></extra>',
        showscale=True
    ))
    
    fig.update_layout(
        title='Feature Correlation Matrix (Sample)',
        height=600,
        template='plotly_white'
    )
    
    return fig


def main():
    # Title
    st.markdown('<p class="main-header">Network Intrusion Detection Dashboard</p>', 
                unsafe_allow_html=True)
    st.markdown("### Compare ML Models for Detecting Network Attacks (UNSW-NB15 Dataset)")
    
    # Connect to database
    try:
        connection = connect_to_db()
    except Exception as e:
        st.error(f"Failed to connect to database: {e}")
        st.info("Please ensure your PostgreSQL database is running and .env file is configured correctly.")
        return
    
    try:
        # Sidebar
        st.sidebar.header("Dashboard Settings")
        
        # Load training runs
        runs_df = load_training_runs(connection)
        
        if len(runs_df) == 0:
            st.error("No training runs found in the database.")
            st.info("Please run `train.py` first to train models and populate the database.")
            return
        
        st.sidebar.success(f"Found {len(runs_df)} training run(s)")
        
        # Model selection
        st.sidebar.subheader("Select Models to Compare")
        selected_runs = []
        
        for idx, row in runs_df.iterrows():
            run_id = row['run_id']
            model_name = row['model_name']
            created_at = row['created_at']
            algorithm = row['algorithm']
            
            label = f"{model_name} ({algorithm})"
            if st.sidebar.checkbox(label, value=True, key=run_id):
                selected_runs.append(run_id)
        
        if len(selected_runs) == 0:
            st.warning("Please select at least one model to display.")
            return
        
        # Filter selected runs
        selected_runs_df = runs_df[runs_df['run_id'].isin(selected_runs)]
        
        # Load metrics for selected runs
        all_metrics = []
        for run_id in selected_runs:
            metrics_df = load_metrics(connection, run_id)
            all_metrics.append(metrics_df)
        metrics_df = pd.concat(all_metrics, ignore_index=True)
        
        # Main content tabs
        tab1, tab2, tab3, tab4 = st.tabs([
            "Model Performance", 
            "ROC & Confusion Matrix", 
            "Data Exploration",
            "Model Details"
        ])
        
        # Tab 1: Model Performance Overview
        with tab1:
            st.header("Model Performance Overview")
            
            # Display metrics in columns
            metric_names = metrics_df['metric_name'].unique()
            
            if len(metric_names) > 0:
                cols = st.columns(len(metric_names))
                
                for idx, metric_name in enumerate(metric_names):
                    with cols[idx]:
                        metric_values = []
                        for run_id in selected_runs:
                            run_metric = metrics_df[
                                (metrics_df['run_id'] == run_id) & 
                                (metrics_df['metric_name'] == metric_name)
                            ]
                            if len(run_metric) > 0:
                                metric_values.append(run_metric['metric_value'].values[0])
                        
                        if metric_values:
                            avg_value = np.mean(metric_values)
                            st.metric(
                                label=metric_name.replace('_', ' ').title(),
                                value=f"{avg_value:.4f}",
                                help=f"Average across {len(metric_values)} model(s)"
                            )
            
            st.markdown("---")
            
            # Performance comparison table
            st.subheader("📋 Detailed Metrics Comparison")
            comparison_data = []
            
            for _, run in selected_runs_df.iterrows():
                run_id = run['run_id']
                model_name = run['model_name']
                run_metrics = metrics_df[metrics_df['run_id'] == run_id]
                
                row = {
                    'Model': model_name,
                    'Algorithm': run['algorithm'],
                    'Run ID': run_id[:8]
                }
                
                for _, metric_row in run_metrics.iterrows():
                    metric_name = metric_row['metric_name'].replace('_', ' ').title()
                    row[metric_name] = f"{metric_row['metric_value']:.4f}"
                
                comparison_data.append(row)
            
            comparison_df = pd.DataFrame(comparison_data)
            st.dataframe(comparison_df, width='stretch', hide_index=True)
            
            st.markdown("---")
            
            # Metrics comparison chart
            st.subheader("Visual Comparison")
            st.plotly_chart(
                plot_metrics_comparison(selected_runs_df, metrics_df),
                width='stretch'
            )
        
        # Tab 2: ROC & Confusion Matrix
        with tab2:
            st.header("ROC Curves & Confusion Matrices")
            
            # ROC Curves
            st.subheader("Receiver Operating Characteristic (ROC) Curves")
            try:
                st.plotly_chart(
                    plot_roc_curve(connection, selected_runs, selected_runs_df),
                    width='stretch'
                )
            except Exception as e:
                st.error(f"Error generating ROC curve: {e}")
            
            st.markdown("---")
            
            # Confusion matrices
            st.subheader("Confusion Matrices")
            
            cols = st.columns(min(len(selected_runs), 2))
            
            for idx, run_id in enumerate(selected_runs):
                with cols[idx % 2]:
                    try:
                        # Load predictions
                        preds_df = load_predictions(connection, run_id)
                        
                        # Load actual labels
                        query = """
                        SELECT r.id, r.label 
                        FROM raw_network_traffic r
                        WHERE r.dataset_split = 'test'
                        ORDER BY r.id
                        """
                        actual_df = fetch_data(connection, query)
                        
                        # Merge
                        merged = preds_df.merge(
                            actual_df, 
                            left_on='record_id', 
                            right_on='id',
                            how='inner'
                        )
                        
                        if len(merged) == 0:
                            st.warning(f"No matching predictions found for this model")
                            continue
                        
                        y_true = merged['label']
                        y_pred = merged['predicted_label']
                        
                        model_name = selected_runs_df[
                            selected_runs_df['run_id'] == run_id
                        ]['model_name'].values[0]
                        
                        st.plotly_chart(
                            plot_confusion_matrix(y_true, y_pred, model_name),
                            width='stretch'
                        )
                    except Exception as e:
                        st.error(f"Error generating confusion matrix: {e}")
            
            st.markdown("---")
            
            # Prediction distributions
            st.subheader("Prediction Probability Distributions")
            
            cols = st.columns(min(len(selected_runs), 2))
            
            for idx, run_id in enumerate(selected_runs):
                with cols[idx % 2]:
                    try:
                        preds_df = load_predictions(connection, run_id)
                        model_name = selected_runs_df[
                            selected_runs_df['run_id'] == run_id
                        ]['model_name'].values[0]
                        
                        st.plotly_chart(
                            plot_prediction_distribution(preds_df, model_name),
                            width='stretch'
                        )
                    except Exception as e:
                        st.error(f"Error generating distribution: {e}")
        
        # Tab 3: Data Exploration
        with tab3:
            st.header("Data Exploration")
            
            # Data overview
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Training Data")
                train_sample = load_raw_data(connection, 'train', limit=1000)
                
                st.metric("Total Samples", f"{len(train_sample):,}")
                
                st.plotly_chart(
                    plot_class_distribution(train_sample, "Training Data Class Distribution"),
                    width='stretch'
                )
                
                with st.expander("View Training Data Sample"):
                    st.dataframe(train_sample.head(100), width='stretch')
            
            with col2:
                st.subheader("Test Data")
                test_sample = load_raw_data(connection, 'test', limit=1000)
                
                st.metric("Total Samples", f"{len(test_sample):,}")
                
                st.plotly_chart(
                    plot_class_distribution(test_sample, "Test Data Class Distribution"),
                    width='stretch'
                )
                
                with st.expander("View Test Data Sample"):
                    st.dataframe(test_sample.head(100), width='stretch')
            
            st.markdown("---")
            
            # Feature statistics
            st.subheader("Feature Statistics")
            
            data_split = st.selectbox("Select Data Split", ["train", "test"])
            data_sample = load_raw_data(connection, data_split, limit=5000)
            
            # Display statistics
            numeric_cols = data_sample.select_dtypes(include=[np.number]).columns
            stats_df = data_sample[numeric_cols].describe().T
            stats_df = stats_df.round(4)
            
            st.dataframe(stats_df, width='stretch')
        
        # Tab 4: Model Details
        with tab4:
            st.header("Model Details & Configuration")
            
            for _, run in selected_runs_df.iterrows():
                with st.expander(f"{run['model_name']} - {run['algorithm'].upper()}", expanded=True):
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.markdown("**Run Information**")
                        st.text(f"Run ID: {run['run_id'][:8]}")
                        st.text(f"Created: {run['created_at']}")
                        st.text(f"Algorithm: {run['algorithm']}")
                    
                    with col2:
                        st.markdown("**Dataset Size**")
                        st.text(f"Training: {run['train_rows']:,} samples")
                        st.text(f"Testing: {run['test_rows']:,} samples")
                    
                    with col3:
                        st.markdown("**Performance Summary**")
                        run_metrics = metrics_df[metrics_df['run_id'] == run['run_id']]
                        
                        for _, metric in run_metrics.iterrows():
                            st.text(f"{metric['metric_name']}: {metric['metric_value']:.4f}")
                    
                    st.markdown("**Hyperparameters**")
                    try:
                        hyperparams = json.loads(run['hyperparameters'])
                        st.json(hyperparams)
                    except:
                        st.text(run['hyperparameters'])
        
        # Footer
        st.markdown("---")
        st.markdown(
            """
            <div style='text-align: center; color: gray;'>
            <p>Network Intrusion Detection System Dashboard | UNSW-NB15 Dataset</p>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    except Exception as e:
        st.error(f"An error occurred: {e}")
        import traceback
        st.code(traceback.format_exc())
    
    finally:
        close_connection(connection)


if __name__ == "__main__":
    main()