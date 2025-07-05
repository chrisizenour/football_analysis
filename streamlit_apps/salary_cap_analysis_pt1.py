# streamlit run streamlit_apps/salary_cap_analysis_pt1.py

import streamlit as st
import joblib
import json
import pandas as pd
import numpy as np
import scipy.stats as stats
from sklearn.tree import export_text
from sklearn.preprocessing import (
    StandardScaler,
    OneHotEncoder,
    MinMaxScaler,
    RobustScaler,
    OrdinalEncoder
)
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.neighbors import KNeighborsRegressor
import matplotlib.pyplot as plt
import seaborn as sns
import plotly_express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import sys
# Ensure the top-level project directory is in the Python path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from path_config import project_path, project_data_sources_path, project_data_exports_path, project_pt_1_models_path, project_papers_path

# Set the page config first before any output
st.set_page_config('NFL Salary Cap Analysis, 2011 - 2024', layout="wide", page_icon=":football:")

@st.cache_data
def load_spotrac_dataset(project_data_exports_path):
    df = pd.read_csv(
        project_data_exports_path / 'spotrac_salary_cap_data_df.csv',
        # sheet_name='Sheet1',
        # header=1,
        # engine='openpyxl',
    )
    # df = df.iloc[:-2]
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])
    dtype_map = {'season': 'int'}
    df = enforce_dtypes(df, dtype_map)
    return df

@st.cache_data
def load_nfl_season_records_dataset(project_data_exports_path):
    df = pd.read_csv(
        project_data_exports_path / 'nfl_season_records_df.csv',
        # sheet_name='Sheet1',
        # header=1,
        # engine='openpyxl',
    )
    # df = df.iloc[:-2]
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])
    dtype_map = {'season': 'int'}
    df = enforce_dtypes(df, dtype_map)
    return df

@st.cache_data
def load_spotrac_nfl_records_dataset(project_data_exports_path):
    df = pd.read_csv(
        project_data_exports_path / 'spotrac_nfl_records_df.csv',
        # sheet_name='Sheet1',
        # header=1,
        # engine='openpyxl',
    )
    # df = df.iloc[:-2]
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])
    dtype_map = {'season': 'int'}
    df = enforce_dtypes(df, dtype_map)
    return df

@st.cache_data
def load_spotrac_nfl_team_season_roster_df_dataset(project_data_exports_path):
    df = pd.read_csv(
        project_data_exports_path / 'spotrac_nfl_team_season_roster_df.csv',
        # sheet_name='Sheet1',
        # header=1,
        # engine='openpyxl',
    )
    # df = df.iloc[:-2]
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])
    dtype_map = {'season': 'int'}
    df = enforce_dtypes(df, dtype_map)
    return df

@st.cache_data
def load_spotrac_nfl_team_season_roster_wide_df_dataset(project_data_exports_path):
    df = pd.read_csv(
        project_data_exports_path / 'spotrac_nfl_team_season_roster_wide_df.csv',
        # sheet_name='Sheet1',
        # header=1,
        # engine='openpyxl',
    )
    # df = df.iloc[:-2]
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])
    dtype_map = {'season': 'int'}
    df = enforce_dtypes(df, dtype_map)
    return df

@st.cache_data
def load_kmc_labeled_df_dataset(project_data_exports_path):
    df = pd.read_csv(
        project_data_exports_path / 'kmc_labeled_df.csv',
        # sheet_name='Sheet1',
        # header=1,
        # engine='openpyxl',
    )
    # df = df.iloc[:-2]
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])
    dtype_map = {'season': 'int'}
    df = enforce_dtypes(df, dtype_map)
    df['label'] = pd.Categorical(df['label'].astype(str))
    return df

@st.cache_data
def load_kmc_grouped_clusters_labeled_df_dataset(project_data_exports_path):
    df = pd.read_csv(
        project_data_exports_path / 'kmc_grouped_clusters_labeled_df.csv',
        # sheet_name='Sheet1',
        # header=1,
        # engine='openpyxl',
    )
    # df = df.iloc[:-2]
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])
    dtype_map = {'season': 'int'}
    df = enforce_dtypes(df, dtype_map)
    return df

@st.cache_data
def load_kmc_grouped_clusters_team_labeled_df_dataset(project_data_exports_path):
    df = pd.read_csv(
        project_data_exports_path / 'kmc_grouped_clusters_team_labeled_df.csv',
        # sheet_name='Sheet1',
        # header=1,
        # engine='openpyxl',
    )
    # df = df.iloc[:-2]
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])
    dtype_map = {'season': 'int'}
    df = enforce_dtypes(df, dtype_map)
    return df

@st.cache_data
def load_gmm_labeled_df_dataset(project_data_exports_path):
    df = pd.read_csv(
        project_data_exports_path / 'gmm_labeled_df.csv',
        # sheet_name='Sheet1',
        # header=1,
        # engine='openpyxl',
    )
    # df = df.iloc[:-2]
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])
    dtype_map = {'season': 'int'}
    df = enforce_dtypes(df, dtype_map)
    df['label'] = pd.Categorical(df['label'].astype(str))
    return df

@st.cache_data
def load_gmm_grouped_clusters_labeled_df_dataset(project_data_exports_path):
    df = pd.read_csv(
        project_data_exports_path / 'gmm_grouped_clusters_labeled_df.csv',
        # sheet_name='Sheet1',
        # header=1,
        # engine='openpyxl',
    )
    # df = df.iloc[:-2]
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])
    dtype_map = {'season': 'int'}
    df = enforce_dtypes(df, dtype_map)
    return df

@st.cache_data
def load_gmm_grouped_clusters_team_labeled_df_dataset(project_data_exports_path):
    df = pd.read_csv(
        project_data_exports_path / 'gmm_grouped_clusters_team_labeled_df.csv',
        # sheet_name='Sheet1',
        # header=1,
        # engine='openpyxl',
    )
    # df = df.iloc[:-2]
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])
    dtype_map = {'season': 'int'}
    df = enforce_dtypes(df, dtype_map)
    return df

@st.cache_data
def load_dbscan_labeled_df_dataset(project_data_exports_path):
    df = pd.read_csv(
        project_data_exports_path / 'dbscan_labeled_df.csv',
        # sheet_name='Sheet1',
        # header=1,
        # engine='openpyxl',
    )
    # df = df.iloc[:-2]
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])
    dtype_map = {'season': 'int'}
    df = enforce_dtypes(df, dtype_map)
    df['label'] = pd.Categorical(df['label'].astype(str))
    return df

@st.cache_data
def load_dbscan_grouped_clusters_labeled_df_dataset(project_data_exports_path):
    df = pd.read_csv(
        project_data_exports_path / 'dbscan_grouped_clusters_labeled_df.csv',
        # sheet_name='Sheet1',
        # header=1,
        # engine='openpyxl',
    )
    # df = df.iloc[:-2]
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])
    dtype_map = {'season': 'int'}
    df = enforce_dtypes(df, dtype_map)
    return df

@st.cache_data
def load_dbscan_grouped_clusters_team_labeled_df_dataset(project_data_exports_path):
    df = pd.read_csv(
        project_data_exports_path / 'dbscan_grouped_clusters_team_labeled_df.csv',
        # sheet_name='Sheet1',
        # header=1,
        # engine='openpyxl',
    )
    # df = df.iloc[:-2]
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])
    dtype_map = {'season': 'int'}
    df = enforce_dtypes(df, dtype_map)
    return df

@st.cache_data
def load_supervised_learning_model_results_pt_1_df_dataset(project_data_exports_path):
    df = pd.read_csv(
        project_data_exports_path / 'model_results_df.csv',
        # sheet_name='Sheet1',
        # header=1,
        # engine='openpyxl',
    )
    # df = df.iloc[:-2]
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])
    dtype_map = {'season': 'int'}
    df = enforce_dtypes(df, dtype_map)
    return df

@st.cache_data
def load_supervised_learning_linear_model_coefficients_pt_1_df_dataset(project_data_exports_path):
    df = pd.read_csv(
        project_data_exports_path / 'linear_model_coefs_df.csv',
        # sheet_name='Sheet1',
        # header=1,
        # engine='openpyxl',
    )
    # df = df.iloc[:-2]
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])
    dtype_map = {'season': 'int'}
    df = enforce_dtypes(df, dtype_map)
    return df

@st.cache_data
def load_X_train_dataset(project_data_exports_path):
    df = pd.read_csv(
        project_data_exports_path / 'X_train_df.csv',
    )
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])
    return df

# load trained models
lr_model_pt_1 = joblib.load(project_pt_1_models_path / 'lr_model.pkl')
tree_model_pt_1 = joblib.load(project_pt_1_models_path / 'tree_model.pkl')
knn_model_pt_1 = joblib.load(project_pt_1_models_path / 'knn_model.pkl')
rf_model_pt_1 = joblib.load(project_pt_1_models_path / 'rf_model.pkl')
ridge_model_pt_1 = joblib.load(project_pt_1_models_path / 'ridge_model.pkl')
lasso_model_pt_1 = joblib.load(project_pt_1_models_path / 'lasso_model.pkl')
elasticnet_model_pt_1 = joblib.load(project_pt_1_models_path / 'elasticnet_model.pkl')
xgbr_model_pt_1 = joblib.load(project_pt_1_models_path / 'xgbr_model.pkl')

def enforce_dtypes(df: pd.DataFrame, dtypes: dict) -> pd.DataFrame:
    for col, dtype in dtypes.items():
        if col in df.columns:
            try:
                df[col] = df[col].astype(dtype)
            except Exception as e:
                st.warning(f"⚠️ Could not convert column `{col}` to `{dtype}`: {e}")
    return df

def summary_stats_df(df, cols):
    summary_stats_results = {}
    for col in cols:
        stats = df[col].agg(['count', 'mean', 'std', 'min', 'var', 'median'])
        quartiles = df[col].quantile([0.25, 0.75])
        summary_stats_results[col] = [stats['count'], stats['mean'], stats['std'], stats['min'], quartiles[0.25], stats['median'], quartiles[0.75], df[col].max(), stats['var']]

    summary_stats_results_df = pd.DataFrame(
        summary_stats_results,
        index=['count', 'mean', 'std', 'min', '25%', 'median', '75%', 'max', 'variance']
    )
    return summary_stats_results_df

def correlation_matrix(df):
    """create correlation matrix dataframe"""
    df_corr_mat = df.corr()
    return df_corr_mat

def correlation_series(df):
    """create a dataframe which has unique feature pairs and their associated correlation coefficient"""
    upper_corr_mat = df.where(np.triu(np.ones(df.shape), k=1).astype(bool))
    unique_corr_pairs = upper_corr_mat.unstack().dropna()
    df_corr_series = unique_corr_pairs.sort_values(ascending=False)
    df_corr_series = df_corr_series.reset_index()
    df_corr_series = df_corr_series.rename(columns={
        'level_0': 'feature_1',
        'level_1': 'feature_2',
        0: 'correlation_coefficient'
    })
    return df_corr_series

def correlation_plot(df, title='Correlation Heatmap'):
    """create a correlation heatmap plot using seaborn"""
    mask = np.triu(np.ones_like(df, dtype=bool))
    f, ax = plt.subplots(1, 1, figsize=(11,9), facecolor='white')
    # cmap = sns.diverging_palette(230, 20, as_cmap=True)
    cmap = 'RdBu'
    sns.heatmap(df,
                mask=mask,
                cmap=cmap,
                vmax=1,
                vmin=-1,
                center=0,
                annot=False,
                square=True,
                linewidths=0.5,
                cbar_kws={'shrink': 0.5})
    ax.set_title(title)
    ax.set_xlabel('Variables')
    ax.set_ylabel('Variables')

    plt.tight_layout()
    return f

def p_val_matrix(df):
    """create p-value matrix dataframe"""
    df_p_val_mat = df.corr(method=lambda x, y: stats.pearsonr(x, y)[1]) - np.eye(len(df.columns))
    return df_p_val_mat

def p_val_series(df):
    """create a dataframe which has unique feature pairs and their associated level of statistical significance"""
    upper_corr_mat = df.where(np.triu(np.ones(df.shape), k=1).astype(bool))
    unique_p_val_pairs = upper_corr_mat.unstack().dropna()
    df_p_val_series = unique_p_val_pairs.sort_values()
    df_p_val_series = df_p_val_series.reset_index()
    df_p_val_series = df_p_val_series.rename(columns={
        'level_0': 'feature_1',
        'level_1': 'feature_2',
        0: 'p_value'
    })
    return df_p_val_series

def p_val_plot(df, title="P-value Heatmap (Green: Significant, White: Not Significant)"):
    """create a level of statistical significance heatmap plot using seaborn"""
    alpha = 0.05
    mask = np.triu(np.ones_like(df, dtype=bool))
    f, ax = plt.subplots(1, 1, figsize=(11,9))
    green = sns.light_palette('seagreen', reverse=True, as_cmap=True)
    green.set_over('white')
    # cmap = sns.diverging_palette(230, 20, as_cmap=True)
    sns.heatmap(df,
                mask=mask,
                cmap=green,
                # cmap=cmap,
                vmax=alpha,
                vmin=0,
                # center=0,
                annot=True,
                square=True,
                linewidths=0.5,
                cbar_kws={'shrink': 0.5})
    ax.set_title(title)
    ax.set_xlabel('Variables')
    ax.set_ylabel('Variables')
    plt.tight_layout()
    return f

def get_transformed_feature_names(model, original_feature_names):
    """
    Extract the transformed feature names from a pipeline's ColumnTransformer.

    Parameters:
    - model: A GridSearchCV object containing a pipeline with a 'preprocessor' step.
    - original_feature_names: List of original feature names before transformation.

    Returns:
    - List of transformed feature names.
    """
    # Access the ColumnTransformer
    preprocessor = model.best_estimator_.named_steps['preprocessor']
    # Get the transformed feature names
    transformed_names = preprocessor.get_feature_names_out(original_feature_names)
    # Clean up the names by removing the transformer prefix (e.g., 'num__' or 'cat__')
    cleaned_names = [name.split('__')[-1] for name in transformed_names]
    return cleaned_names


def extract_model_info(model, model_name, original_feature_names):
    """
    Extract coefficients, intercept, decision tree structure, feature importances, and hyperparameters from a model.

    Parameters:
    - model: A GridSearchCV object containing a pipeline.
    - model_name: Shorthand name of the model (e.g., 'lr', 'tree').
    - original_feature_names: List of original feature names.

    Returns:
    - Dictionary containing the model information.
    """
    info = {}

    # Access the best estimator (pipeline) and regressor
    pipeline = model.best_estimator_
    regressor = pipeline.named_steps['regressor']

    # Get transformed feature names
    transformed_features = get_transformed_feature_names(model, original_feature_names)

    # Extract coefficients and intercept for Linear Regression, Ridge, Lasso, ElasticNet
    if model_name in ["lr", "ridge", "lasso", "elasticnet"]:
        coefs = regressor.coef_
        # Initialize the coefficients dictionary with feature coefficients
        coefficients_dict = {feature: float(coef) for feature, coef in zip(transformed_features, coefs)}
        # Add the intercept to the dictionary
        intercept = float(regressor.intercept_)
        coefficients_dict["Intercept"] = intercept
        info["coefficients"] = coefficients_dict

    # Extract decision tree structure and feature importances for Decision Tree
    if model_name == "tree":
        # Decision tree structure as text
        tree_text = export_text(regressor, feature_names=transformed_features)
        info["decision_tree"] = tree_text
        # Feature importances
        importances = regressor.feature_importances_
        info["feature_importances"] = {feature: float(imp) for feature, imp in zip(transformed_features, importances)}

    # Extract feature importances for Random Forest and XGBoost
    if model_name in ["rf", "xgbr"]:
        importances = regressor.feature_importances_
        info["feature_importances"] = {feature: float(imp) for feature, imp in zip(transformed_features, importances)}

    # Extract optimized hyperparameters from GridSearchCV
    info["best_params"] = model.best_params_

    return info


def summarize_model_preprocessors(models_dict):
    """
    Inspect each model in the dictionary and summarize whether it includes a preprocessor,
    and if so, what transformations are applied to which features.

    Adds a flag for models that *should* have preprocessing (e.g., Ridge, Lasso, KNN) but don't.

    Returns:
    - pd.DataFrame with columns:
        - Model
        - Has Preprocessor
        - Preprocessor Type
        - Scaled Features
        - Encoded Features
        - Needs Preprocessor (Based on model type)
        - Warning (if expected but missing)
    """
    import pandas as pd

    summary = []

    for name, model in models_dict.items():
        pipeline = None
        if hasattr(model, 'best_estimator_'):
            pipeline = model.best_estimator_
        elif isinstance(model, Pipeline):
            pipeline = model

        regressor = None
        if pipeline and hasattr(pipeline, 'named_steps') and 'regressor' in pipeline.named_steps:
            regressor = pipeline.named_steps['regressor']
        elif hasattr(model, 'predict') and not hasattr(model, 'named_steps'):
            regressor = model  # Not a pipeline, but standalone model

        # Determine if model type typically requires preprocessing
        needs_preprocessing = isinstance(regressor, (Ridge, Lasso, ElasticNet, KNeighborsRegressor))

        has_preprocessor = False
        preprocessor_type = "None"
        scaled_features = []
        encoded_features = []

        if pipeline and hasattr(pipeline, 'named_steps') and 'preprocessor' in pipeline.named_steps:
            preprocessor = pipeline.named_steps['preprocessor']
            preprocessor_type = type(preprocessor).__name__

            if hasattr(preprocessor, 'transformers'):
                for transformer_name, transformer_obj, columns in preprocessor.transformers:
                    if transformer_name == 'remainder':
                        continue  # skip passthrough or drop
                    if hasattr(transformer_obj, 'get_params'):
                        if 'scaler' in transformer_name or isinstance(transformer_obj, StandardScaler):
                            scaled_features.extend(columns)
                        elif 'encoder' in transformer_name or isinstance(transformer_obj, OneHotEncoder):
                            encoded_features.extend(columns)
                        else:
                            if 'scale' in str(transformer_obj).lower():
                                scaled_features.extend(columns)
                            if 'onehot' in str(transformer_obj).lower():
                                encoded_features.extend(columns)
            has_preprocessor = bool(scaled_features or encoded_features)

        warning = ""
        if needs_preprocessing and not has_preprocessor:
            warning = "⚠️ Needs preprocessing but none found"

        summary.append({
            "Model": name,
            "Has Preprocessor": has_preprocessor,
            "Preprocessor Type": preprocessor_type,
            "Scaled Features": ', '.join(scaled_features) if scaled_features else "-",
            "Encoded Features": ', '.join(encoded_features) if encoded_features else "-",
            "Needs Preprocessor": needs_preprocessing,
            "Warning": warning
        })

    return pd.DataFrame(summary)


def main():
    # st.set_page_config('NFL Salary Cap Analysis, 2011 - 2024', layout="wide", page_icon=":football:")
    st.markdown('# NFL Salary Cap Analysis, 2011 - 2024')
    st.markdown("""
    Data sourced from: 
    
    Spotrac.com (https://www.spotrac.com/nfl/{team}/overview/_/year/{season}/sort/cap_total) 
    
    NFL.com (https://www.nfl.com/standings/league/{season}/REG)
    """)
    st.markdown("## Is Team Performance Influenced by Annual Salary Cap Distributions?")

    # --- load datasets ---
    X_train = load_X_train_dataset(project_data_exports_path)

    spotrac_salary_cap_data_df = load_spotrac_dataset(project_data_exports_path)
    nfl_season_records_df = load_nfl_season_records_dataset(project_data_exports_path)
    spotrac_nfl_records_df = load_spotrac_nfl_records_dataset(project_data_exports_path)
    spotrac_nfl_team_season_roster_df = load_spotrac_nfl_team_season_roster_df_dataset(project_data_exports_path)
    spotrac_nfl_team_season_roster_wide_df = load_spotrac_nfl_team_season_roster_wide_df_dataset(project_data_exports_path)

    kmc_labeled_df = load_kmc_labeled_df_dataset(project_data_exports_path)
    kmc_grouped_clusters_labeled_df = load_kmc_grouped_clusters_labeled_df_dataset(project_data_exports_path)
    kmc_grouped_clusters_team_labeled_df = load_kmc_grouped_clusters_team_labeled_df_dataset(project_data_exports_path)

    gmm_labeled_df = load_gmm_labeled_df_dataset(project_data_exports_path)
    gmm_grouped_clusters_labeled_df = load_gmm_grouped_clusters_labeled_df_dataset(project_data_exports_path)
    gmm_grouped_clusters_team_labeled_df = load_gmm_grouped_clusters_team_labeled_df_dataset(project_data_exports_path)

    dbscan_labeled_df = load_dbscan_labeled_df_dataset(project_data_exports_path)
    dbscan_grouped_clusters_labeled_df = load_dbscan_grouped_clusters_labeled_df_dataset(project_data_exports_path)
    dbscan_grouped_clusters_team_labeled_df = load_dbscan_grouped_clusters_team_labeled_df_dataset(project_data_exports_path)

    supervised_learning_pt_1_model_results_df = load_supervised_learning_model_results_pt_1_df_dataset(project_data_exports_path)
    supervised_learning_pt_1_linear_model_coefs_df = load_supervised_learning_linear_model_coefficients_pt_1_df_dataset(project_data_exports_path)

    lit_review, tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
        'Literature Review',
        'Spotrac Data',
        'Team Record Data',
        'Spotrac Data + Team Record Data',
        'EDA - Pt 1',
        'Unsupervised Learning - Pt 1',
        'Supervised Learning - Pt 1',
        'Predictive Modeling - Pt 1',
        'Takeaways - Pt 1, Way Ahead for Pt 2'
    ])

    with lit_review:
        # Load and display Markdown file
        md_path = Path(project_papers_path) / "literature_review.md"
        try:
            with open(md_path, "r", encoding="utf-8") as f:
                full_markdown = f.read()
            st.markdown(full_markdown, unsafe_allow_html=True)
        except FileNotFoundError:
            st.error(f"Markdown file not found at {md_path}. Please ensure it was generated correctly.")

        # Interactive Table 1
        st.subheader("Table 1: Summary of Cited Studies")
        table_data = pd.DataFrame({
            "Study": ["Leeds & Kowalewski (2001)", "Borghesi (2008)", "Mondello & Maxcy (2009)",
                      "Mulholland & Jensen (2019)", "Keefer (2017)", "Shin et al. (2023)"],
            "Data Scope": ["~500 players, 1992–1994", "~352 team-seasons, 1994–2004",
                           "~256 team-seasons, 2000–2007", "~2,500 players, 2011–2015",
                           "~1,000–2,000 players, 2004–2012", "~660 MLB team-seasons, 2001–2022"],
            "Methodology": ["Quantile regression", "OLS, Poisson regression", "OLS regression",
                            "Regression, linear programming", "OLS regression", "GMM panel regression"],
            "Key Findings": [
                "Performance drives salaries; income inequality rises",
                "Lower dispersion improves wins",
                "High dispersion reduces wins; bonuses help QBs",
                "Prioritize DEs, OLBs; rookie wins key",
                "Sunk-cost retention harms performance",
                "New hire budgets harm; star pay helps but weakens"
            ],
            "Limitations": [
                "Skill position focus; two-year scope",
                "Team-level focus; pre-2011 data",
                "Team-level focus; outdated data",
                "Assumes market efficiency; no cohesion focus",
                "Pre-2013 data; no position focus",
                "MLB context; no perception data"
            ]
        })
        st.dataframe(table_data, use_container_width=True)

        # Collapsible Study Details
        # st.subheader("Study Details")
        # with st.expander("Leeds & Kowalewski (2001)"):
        #     st.markdown(
        #         "Quantile regression on ~500 skill position players (1992–1994) showed post-CBA income inequality, with performance driving salaries. Limited to skill positions.")
        # with st.expander("Borghesi (2008)"):
        #     st.markdown(
        #         "OLS/Poisson regression on ~352 team-seasons (1994–2004) found lower dispersion improves wins via team cohesion.")
        # with st.expander("Mondello & Maxcy (2009)"):
        #     st.markdown(
        #         "OLS on ~256 team-seasons (2000–2007) showed high dispersion harms wins; bonuses help QBs. Outdated data.")
        # with st.expander("Mulholland & Jensen (2019)"):
        #     st.markdown(
        #         "Regression/linear programming on ~2,500 players (2011–2015) prioritized DEs (13.7%), OLBs (15.2%). Assumes market efficiency, lacks cohesion focus.")
        # with st.expander("Keefer (2017)"):
        #     st.markdown(
        #         "OLS on ~1,000–2,000 players (2004–2012) showed sunk-cost retention harms performance. Pre-2013 data.")
        # with st.expander("Shin et al. (2023)"):
        #     st.markdown(
        #         "GMM on ~660 MLB team-seasons (2001–2022) found new hire budgets harm performance; star pay helps but weakens in top teams. MLB context.")

        # Tooltips for key terms
        st.info("**RBV**: Resource-Based View, a framework for leveraging unique resources for competitive advantage.")
        st.info("**AV**: Approximate Value, a Pro-Football-Reference metric for player performance.")

    with tab1:
        st.markdown("#### Spotrac Data")
        st.dataframe(spotrac_salary_cap_data_df)
        st.write(f'Number of Observations: {spotrac_salary_cap_data_df.shape[0]}')
        st.write(f'Max number of tables: {spotrac_salary_cap_data_df['table_number'].max()}')

        spotrac_salary_team_salary_cap_pct_df = spotrac_salary_cap_data_df.groupby(['team', 'season'], observed=True).agg({'cap_hit_pct_league_cap': 'sum'}).reset_index()
        # st.dataframe(spotrac_salary_team_salary_cap_pct_df)

        spotrac_salary_overall_salary_cap_pct_df = spotrac_salary_team_salary_cap_pct_df.groupby(['season'], observed=True).agg({'cap_hit_pct_league_cap': 'mean'}).reset_index()
        # st.dataframe(spotrac_salary_overall_salary_cap_pct_df)

        # Multiselect for teams
        selected_teams = st.multiselect(
            "Select Teams to Highlight",
            options=spotrac_salary_team_salary_cap_pct_df['team'].unique(),
            default=[]
        )

        # Initialize figure
        spotrac_cap_hit_pct_plot = go.Figure()

        # Add team lines
        for team in spotrac_salary_team_salary_cap_pct_df['team'].unique():
            team_df = spotrac_salary_team_salary_cap_pct_df[
                spotrac_salary_team_salary_cap_pct_df['team'] == team
                ]

            line_color = 'gray' if team not in selected_teams else None
            line_width = .5 if team not in selected_teams else 4

            spotrac_cap_hit_pct_plot.add_trace(go.Scatter(
                x=team_df['season'],
                y=team_df['cap_hit_pct_league_cap'],
                mode='lines',
                name=team,
                line=dict(color=line_color, width=line_width),
                opacity=0.5 if team not in selected_teams else 1.0,
                showlegend=team in selected_teams  # only show legend for selected teams
            ))

        # Add overall average line
        spotrac_cap_hit_pct_plot.add_trace(go.Scatter(
            x=spotrac_salary_overall_salary_cap_pct_df['season'],
            y=spotrac_salary_overall_salary_cap_pct_df['cap_hit_pct_league_cap'],
            mode='lines+markers',
            name='League Avg',
            line=dict(
                color='black',
                width=4,
                dash='dash'
            ),
            marker=dict(size=6),
            showlegend=True
        ))

        # Update layout
        spotrac_cap_hit_pct_plot.update_layout(
            title=f"Sum of `cap_hit_pct_league_cap` Salary Cap by Team per Season",
            xaxis_title="NFL Season",
            yaxis_title="Team Salary Cap %",
            height=600,
            margin=dict(r=120)
        )

        st.plotly_chart(spotrac_cap_hit_pct_plot, use_container_width=True)

        spotrac_salary_team_roster_status_salary_cap_pct_df = \
        spotrac_salary_cap_data_df.groupby(['team', 'season', 'roster_status'], observed=True).agg({'cap_hit_pct_league_cap': 'sum'}).reset_index()
        # st.dataframe(spotrac_salary_team_roster_status_salary_cap_pct_df)

        spotrac_salary_overall_roster_status_salary_cap_pct_df = \
            spotrac_salary_team_roster_status_salary_cap_pct_df.groupby(['season', 'roster_status'], observed=True).agg({'cap_hit_pct_league_cap': 'mean'}).reset_index()
        # st.dataframe(spotrac_salary_overall_roster_status_salary_cap_pct_df)

        # Initialize figure
        spotrac_roster_status_cap_hit_pct_plot = go.Figure()

        # Add team lines by roster_status
        for team in spotrac_salary_team_roster_status_salary_cap_pct_df['team'].unique():
            for roster_status in ['active', 'inactive']:
                subset_df = spotrac_salary_team_roster_status_salary_cap_pct_df[
                    (spotrac_salary_team_roster_status_salary_cap_pct_df['team'] == team) &
                    (spotrac_salary_team_roster_status_salary_cap_pct_df['roster_status'] == roster_status)
                    ]

                if subset_df.empty:
                    continue

                line_color = 'gray' if team not in selected_teams else None
                line_width = 1 if team not in selected_teams else 3
                opacity = 0.4 if team not in selected_teams else 1.0
                dash_style = 'solid' if roster_status == 'active' else 'dot'

                spotrac_roster_status_cap_hit_pct_plot.add_trace(go.Scatter(
                    x=subset_df['season'],
                    y=subset_df['cap_hit_pct_league_cap'],
                    mode='lines',
                    name=f"{team} ({roster_status})",
                    line=dict(color=line_color, width=line_width, dash=dash_style),
                    opacity=opacity,
                    showlegend=team in selected_teams
                ))

        # Add overall average lines per roster_status
        for roster_status in ['active', 'inactive']:
            overall_df = spotrac_salary_overall_roster_status_salary_cap_pct_df[
                spotrac_salary_overall_roster_status_salary_cap_pct_df['roster_status'] == roster_status
                ]

            spotrac_roster_status_cap_hit_pct_plot.add_trace(go.Scatter(
                x=overall_df['season'],
                y=overall_df['cap_hit_pct_league_cap'],
                mode='lines+markers',
                name=f'League Avg ({roster_status})',
                line=dict(
                    color='black',
                    width=4,
                    dash='solid' if roster_status == 'active' else 'dash'
                ),
                marker=dict(size=6),
                showlegend=True
            ))

        # Layout updates
        spotrac_roster_status_cap_hit_pct_plot.update_layout(
            title=f"Sum of `cap_hit_pct_league_cap` by Team and Roster Status per Season",
            xaxis_title="NFL Season",
            yaxis_title="Team Salary Cap %",
            height=650,
            margin=dict(r=120)
        )

        st.plotly_chart(spotrac_roster_status_cap_hit_pct_plot, use_container_width=True)

        with st.expander("Spotrac Summary Stats Tables"):
            tab1col1, tab1col2 = st.columns([.4, .6])
            with tab1col1:
                st.write("Overall Spotrac Dataset")
                st.dataframe(summary_stats_df(spotrac_salary_cap_data_df, ['cap_hit', 'cap_hit_pct_league_cap']))
            with tab1col2:
                roster_status_list = ['active', 'inactive']
                spotrac_summary_dfs = []
                for roster_status in roster_status_list:
                    roster_status_df = spotrac_salary_cap_data_df[spotrac_salary_cap_data_df['roster_status'] == roster_status]
                    roster_status_summary = summary_stats_df(roster_status_df, ['cap_hit', 'cap_hit_pct_league_cap'])
                    # Add MultiIndex column: (variable, roster_status)
                    roster_status_summary.columns = pd.MultiIndex.from_product(
                        [roster_status_summary.columns, [roster_status]]
                    )
                    spotrac_summary_dfs.append(roster_status_summary)
                # Combine and flatten MultiIndex columns
                spotrac_summary_stats_by_roster_status_df = pd.concat(spotrac_summary_dfs, axis=1)
                spotrac_summary_stats_by_roster_status_df.columns = [
                    f"{col}_{status}" for col, status in spotrac_summary_stats_by_roster_status_df.columns
                ]
                st.write("By Roster Status Spotrac Dataset")
                st.dataframe(spotrac_summary_stats_by_roster_status_df)
        with st.expander("Future uses of this dataset"):
            st.write("""
            For this analysis, I will use the following columns:
            - `cap_hit`: This represents the dollar amount spent by teams on players that counts toward their salary cap.
            - `pos`: This represents the position of the player.
            - `team`: This represents the team that the player plays for.
            - `season`: This represents the year of the season.
            - `table_number`: This represents the number of the table from Spotrac the player is on.
                - table 0 is the active roster
                - table 1 and greater is the inactive roster
            - `position_level_one`: This is an engineered column the represents the highest positional grouping of the player (offense, defense, special team)
            - `position_level_two`: This is an engineered column the represents the second highest positional grouping of the player (offensive_line, defensive_line, running_back, etc.)
            
            - `cap_hit_pct_league_cap`: This represents the percentage of the salary cap that the player consumed of that season's team salary cap.
                - When summed, teams can have greater than 100% of the season's salary cap.
                - Multiple instances of teams with greater than 100% of the salary cap are possible.
                - Follow-on analysis will use the `cap_hit` value and consider the each team's salary cap allocations to be 100% and the constituent proportions of the salary cap as the cap hit percentages.
            """)

    with tab2:
        st.markdown("#### Team Record Data")
        st.dataframe(nfl_season_records_df)
        st.write(f'Number of Observations: {nfl_season_records_df.shape[0]}')

        # Fix groupby output to flatten column names
        nfl_season_win_pct_means = (
            nfl_season_records_df.groupby('season', observed=True)['pct']
            .agg(count='count', mean='mean', std='std')
            .reset_index()
            .round({'mean': 8, 'std': 8})
        )

        nfl_win_pct_boxplots = px.violin(
            nfl_season_records_df,
            x='season',
            y='pct',
            title="Winning % per NFL Season",
            labels={
                'season': 'NFL Season',
                'pct': 'Winning %'
            },
            points='all',
            box=True,
        )
        # Add means as scatter trace
        nfl_win_pct_boxplots.add_trace(go.Scatter(
            x=nfl_season_win_pct_means['season'],
            y=nfl_season_win_pct_means['mean'],
            mode='lines+markers',
            marker=dict(symbol='circle', size=10, color='red'),
            name='Mean',
            showlegend=True
        ))
        st.plotly_chart(nfl_win_pct_boxplots, use_container_width=True)

        with st.expander("View Season Win % Means Dataframe"):
            tab2col1, tab2col2 = st.columns([0.2, 0.8])
            with tab2col1:
                st.write("Overall NFL Records Dataset")
                st.dataframe(summary_stats_df(nfl_season_records_df, ['pct']))
            with tab2col2:
                seasons = list(range(2011, 2025))
                nfl_summary_dfs = []
                for season in seasons:
                    season_df = nfl_season_records_df[nfl_season_records_df['season'] == season]
                    nfl_season_summary = summary_stats_df(season_df, ['pct'])
                    nfl_season_summary.columns = [season]
                    nfl_summary_dfs.append(nfl_season_summary)
                nfl_summary_stats_by_season_df = pd.concat(nfl_summary_dfs, axis=1)
                st.write("By Season NFL Records Dataset")
                st.dataframe(nfl_summary_stats_by_season_df)
            # st.write("""
            # - Season-to-season, there is stability in team performance as measured by `pct`
            # - Individual seasons show little variation and are emblematic of the overall dataset
            # - Given consistent nature of `pct`:
            #     - What effects do changing salary cap allocations have on team performance as measured by `pct`?
            #
            # """)

        with st.expander("Future uses of this dataset"):
            st.write("""
            For this analysis, I will use the following columns:
            - `nfl_team_name`: This represents the name of the team.
            - Team outcome (performance) metrics:
                - `pct`: This represents the winning % of the team.
                - `w`: This represents the number of wins of the team.
                - `l`: This represents the number of losses of the team.
                - `pf`: This represents the number of points for of the team.
                - `pa`: This represents the number of points against of the team.
                - `net_pts`: This represents the net points scored by the team.
                - `div_win_pct`: This represents the winning % of the team in the division.
                - `conf_win_pct`: This represents the winning % of the team in the conference.
            - This analysis's initial line of effort will focus on the team's overall season winning percentage (`pct`).
            - Future iterations of analysis may focus on other team performance metrics.
            - Though `pct` is target variable of this analysis, other team performance metrics will be analyzed as features that provide additional information and context
            
            """)



    with tab3:
        st.markdown("#### Spotrac Data + Team Record Data")
        st.write("Merge Spotrac Data and NFL Season Records Data")
        st.dataframe(spotrac_nfl_records_df)
        st.write(f'Number of Observations: {spotrac_nfl_records_df.shape[0]}')

    with tab4:
        st.markdown("#### Team - Season - Roster Status EDA")
        with st.expander("View Dataframe"):
            st.dataframe(spotrac_nfl_team_season_roster_df)
            st.write(f'Number of Observations: {spotrac_nfl_team_season_roster_df.shape[0]}')
            st.write('---')
            st.write("Wide version of `spotrac_nfl_team_season_roster_df`")
            st.dataframe(spotrac_nfl_team_season_roster_wide_df)
            st.write(f'Number of Observations: {spotrac_nfl_team_season_roster_wide_df.shape[0]}')
            st.write("""
            #### Dataframe Metric Descriptions
            - `roster_status`:
                - Indicates roster portion - **active** or **inactive**.
                - Derived from original table source: table 0 = active, table 1+ = inactive.
            
            ##### Core Metrics (Per Team, Season, and Roster Status)
            - `player_count_active`; `player_count_inactive`:
                - Total number of players in each roster status group.
            
            - `cap_hit_sum_active`; `cap_hit_sum_inactive`:
                - Total salary cap hit assigned to active and inactive players.
            
            - `player_count_prop_active`; `player_count_prop_inactive`:
                - Proportion of team’s players who are active or inactive.
                - E.g., 0.85 means 85% of players were active that season.
            
            - `cap_hit_prop_active`; `cap_hit_prop_inactive`:
                - Proportion of salary cap dollars spent on each roster group.
                - E.g., 0.92 means 92% of salary cap went to active players.
            
            - `cap_hit_per_player_prop_active`; `cap_hit_per_player_prop_inactive`:
                - This is a **relative metric** that compares the average cap hit per player **within a roster group** (active/inactive)
                to the **overall average cap hit per player for the entire team** in that season.
                - **Formula**:  
                  `cap_hit_per_player_prop = (cap_hit_sum / player_count) / (total_cap_hit_sum / total_player_count)`
                - **How to interpret**:
                    - `> 1.0` → Players in this group (active/inactive) have a **higher average cap hit per player** than the team average.
                    - `< 1.0` → Players in this group have a **lower average cap hit per player** than the team average.
                    - This helps identify if a team’s money is concentrated among a smaller set of players or more evenly distributed.
            
            ##### Derived Metrics: Comparing Active vs Inactive Rosters

            The following metrics summarize how **disparate or balanced** the two roster groups are.
            
            - `delta_*` metrics (e.g., `delta_cap_hit_prop`):
                - **Definition**: Difference between the active and inactive values for a given metric.
                - **Formula**: `active - inactive`
                - **Interpretation**:
                    - Positive values: active group has a higher value.
                    - Negative values: inactive group has a higher value.
                    - **Use case**: Highlights where resources or player representation are concentrated.
            
            - `ratio_*` metrics (e.g., `ratio_player_count_prop`):
                - **Definition**: Ratio of active value to inactive value.
                - **Formula**: `active / (inactive + epsilon)`
                - **Interpretation**:
                    - `> 1.0`: active value is greater than inactive.
                    - `< 1.0`: inactive value is greater.
                    - `≈ 1.0`: values are roughly equal.
                    - **Use case**: Easy-to-read multiplicative relationship.
                    - Example:
                        - `0.80 active / 0.20 inactive = 4.0` → 4x more active players.
                        - `0.25 active / 0.75 inactive = 0.33` → Active group is one-third the size of inactive.
            
            - `total_*` metrics (e.g., `total_cap_hit_prop`):
                - **Definition**: Sum of the active and inactive values.
                - **Formula**: `active + inactive`
                - **Interpretation**:
                    - These values should typically ≈ 1.0 for proportion metrics (e.g., `cap_hit_prop`), acting as a quick data integrity check.
                    - Deviations from 1.0 could indicate rounding or missing data.
            
            - `log_ratio_*` metrics (e.g., `log_ratio_cap_hit_per_player_prop`):
                - **Definition**: Natural logarithm of the active-to-inactive ratio.
                - **Formula**: `ln(active / inactive)`
                - **Interpretation of log_ratio values**:
                    - `> 0`: active value is greater than inactive.
                    - `< 0`: inactive value is greater.
                    - `= 0`: values are equal.
                    - **Use case**: Especially useful in statistical modeling, as log-transformed ratios handle skew and create symmetry for relative comparisons.
                    - A value of `ln(2) ≈ 0.693` means the active metric is **twice** as large as the inactive one.
            
            ##### Team Performance Metrics (Merged from `nfl_season_records_df`)
            - `w`; `l`:
                - Number of wins and losses in the season.
            
            - `pct`:
                - Win percentage.
            
            - `pf`; `pa`:
                - Points For and Points Against.
            
            - `net_pts`:
                - Point differential: `pf - pa`.
            
            - `div_win_pct`:
                - Division win percentage.
            
            - `conf_win_pct`:
                - Conference win percentage.
            """)

        with st.expander("Correlation Amongst Features"):
            # st.dataframe(spotrac_nfl_team_season_roster_wide_df[['season', 'player_count_prop_active', 'cap_hit_prop_active']].corr())

            corr_mat_df_pt_1 = correlation_matrix(spotrac_nfl_team_season_roster_wide_df[['season',
                                                                                          'player_count_prop_active',
                                                                                          'cap_hit_prop_active',
                                                                                          'cap_hit_per_player_prop_active',
                                                                                          'delta_player_count_prop',
                                                                                          'ratio_player_count_prop',
                                                                                          'log_ratio_player_count_prop',
                                                                                          'delta_cap_hit_prop',
                                                                                          'ratio_cap_hit_prop',
                                                                                          'log_ratio_cap_hit_prop',
                                                                                          'delta_cap_hit_per_player_prop',
                                                                                          'ratio_cap_hit_per_player_prop',
                                                                                          'log_ratio_cap_hit_per_player_prop',
                                                                                          'pct']])
            pval_mat_df_pt_1 = p_val_matrix(spotrac_nfl_team_season_roster_wide_df[['season',
                                                                                          'player_count_prop_active',
                                                                                          'cap_hit_prop_active',
                                                                                          'cap_hit_per_player_prop_active',
                                                                                          'delta_player_count_prop',
                                                                                          'ratio_player_count_prop',
                                                                                          'log_ratio_player_count_prop',
                                                                                          'delta_cap_hit_prop',
                                                                                          'ratio_cap_hit_prop',
                                                                                          'log_ratio_cap_hit_prop',
                                                                                          'delta_cap_hit_per_player_prop',
                                                                                          'ratio_cap_hit_per_player_prop',
                                                                                          'log_ratio_cap_hit_per_player_prop',
                                                                                          'pct']])

            corr_col1, corr_col2 = st.columns(2)
            with corr_col1:
                corr_mat_series_df_pt_1 = correlation_series(corr_mat_df_pt_1)
                pval_mat_series_df_pt_1 = p_val_series(pval_mat_df_pt_1)

                # st.dataframe(corr_mat_series_df_pt_1.style.format({'correlation_coefficient': '{:.5f}'}))
                # st.dataframe(pval_mat_series_df_pt_1)
                corr_mat_p_val_df_merged_pt_1 = pd.merge(corr_mat_series_df_pt_1, pval_mat_series_df_pt_1, left_on=['feature_1', 'feature_2'], right_on=['feature_1', 'feature_2'], how='left')
                st.dataframe(corr_mat_p_val_df_merged_pt_1.style.format({
                    'correlation_coefficient': '{:.5f}',
                    'p_value': '{:.5e}'  # Ensure p_value is displayed in scientific notation
                }))
            with corr_col2:
                corr_mat_plot_pt_1 = correlation_plot(corr_mat_df_pt_1)
                st.pyplot(corr_mat_plot_pt_1, use_container_width=True)
            st.markdown("""
            ## 🔍 Feature Correlation Insights

            This section summarizes key insights from the correlation analysis of NFL team-season features, focusing on player count and salary cap distribution.

            ---

            ### 1. Redundant Features
            - Many features like `delta_*`, `log_ratio_*`, and `ratio_*` are **highly correlated (ρ > 0.99)** with their base counterparts.
            - ✅ **Action:** Keep only one version of each feature (e.g., raw, delta, or log-ratio) to reduce redundancy and avoid multicollinearity.

            ---

            ### 2. What Drives Team Success?
            **Features most correlated with winning percentage (`pct`):**
            - `log_ratio_cap_hit_per_player_prop`: **ρ = 0.51**
            - `cap_hit_prop_active`: **ρ = 0.49**
            - `delta_cap_hit_prop`: **ρ = 0.49**
            - `ratio_cap_hit_per_player_prop`: **ρ = 0.47**

            **Interpretation:** Teams that spend a higher share of their cap on **fewer high-value active players** tend to win more.

            ---

            ### 3. League Strategy Trends Over Time
            - Several features show **strong negative correlation with `season`**, such as:
              - `player_count_prop_active`: **ρ = -0.71**
              - `cap_hit_prop_active`: **ρ = -0.40**

            **Interpretation:** Teams are **consolidating spending on fewer active players** over time, possibly favoring star power or roster efficiency.

            ---

            ### 4. Features With Little to No Predictive Value
            - Some features have low correlation and are **not statistically significant**, such as:
              - `pct` vs. `season`: **ρ ≈ 0**
              - `ratio_player_count_prop` vs. `pct`: **ρ ≈ 0.03**

            **Action:** These can be excluded from your predictive models or explored further with non-linear techniques.

            ---

            ### Recommendations for Modeling
            - **Prioritize** features like `cap_hit_prop_active`, `cap_hit_per_player_prop_active`, and their transformations.
            - **Remove** highly collinear feature variants.
            - **Control for `season`** to account for strategic evolution across years.
            - **Consider interactions** between cap allocation and roster structure.

            ---
            """)

        correlation_plot_features = ['season',
                                     'player_count_prop_active',
                                     'cap_hit_prop_active',
                                     'cap_hit_per_player_prop_active',
                                     'delta_player_count_prop',
                                     'ratio_player_count_prop',
                                     'log_ratio_player_count_prop',
                                     'delta_cap_hit_prop',
                                     'ratio_cap_hit_prop',
                                     'log_ratio_cap_hit_prop',
                                     'delta_cap_hit_per_player_prop',
                                     'ratio_cap_hit_per_player_prop',
                                     'log_ratio_cap_hit_per_player_prop',
                                     'pct']
        correlation_plot_categorical_features = ['team']
        correlation_plot_color_cols = correlation_plot_features + correlation_plot_categorical_features
        corr_plot_col1, corr_plot_col2, corr_plot_col3 = st.columns(3)
        with corr_plot_col1:
            corr_plot_x_col = st.selectbox('Select X-Axis Column', options=correlation_plot_features,
                                           index=correlation_plot_features.index('delta_player_count_prop'), key='correlation_plot_x_axis')
        with corr_plot_col2:
            corr_plot_y_col = st.selectbox('Select Y-Axis Column', options=correlation_plot_features,
                                           index=correlation_plot_features.index('player_count_prop_active'), key='correlation_plot_y_axis')
        with corr_plot_col3:
            corr_plot_color_col = st.selectbox('Select Color', options=correlation_plot_color_cols,
                                         index=correlation_plot_color_cols.index('pct'), key='corr_plot_color')
        if not spotrac_nfl_team_season_roster_wide_df.empty:
            is_discrete = corr_plot_color_col in correlation_plot_categorical_features or (corr_plot_color_col == 'season') or \
                          kmc_labeled_df[corr_plot_color_col].nunique() < 10
            corr_plot_color_param = dict(
                color=corr_plot_color_col,
                color_discrete_sequence=px.colors.qualitative.Plotly if is_discrete else None,
                color_continuous_scale=None if is_discrete else 'Viridis'
            )

        correlation_scatterplot = px.scatter(
            spotrac_nfl_team_season_roster_wide_df,
            x=corr_plot_x_col,
            y=corr_plot_y_col,
            **corr_plot_color_param,
            hover_data=['team', 'season'],
            opacity=0.7,
            size_max=10
        )
        # Update layout
        correlation_scatterplot.update_layout(
            title=f"Scatter Plot: {corr_plot_x_col} vs {corr_plot_y_col} (Colored by {corr_plot_color_col})",
            xaxis_title=corr_plot_x_col.replace('_', ' ').title(),
            yaxis_title=corr_plot_y_col.replace('_', ' ').title(),
            legend_title=corr_plot_color_col.replace('_', ' ').title(),
            template='plotly_white',
            height=600,
            showlegend=True
        )

        st.plotly_chart(correlation_scatterplot, use_container_width=True)

        with st.expander("Cap Hit Salary Proportion Plots"):
            # Plotly boxplot
            overall_season_roster_status_cap_hit_prop_boxplot = px.box(
                spotrac_nfl_team_season_roster_df,
                x='season',
                y='cap_hit_prop',
                color='roster_status',
                title="Proportion of Salary Cap by Roster Status per Season",
                labels={
                    'season': 'NFL Season',
                    'cap_hit_prop': 'Proportion of Team Salary Cap',
                    'roster_status': 'Roster Status'
                }
            )

            # Move legend outside of lineplot
            overall_season_roster_status_cap_hit_prop_boxplot.update_layout(
                legend=dict(
                    title='Roster Status',
                    x=1.05,
                    y=1
                ),
                margin=dict(r=150),  # Create space for legend
                xaxis_title='NFL Season',
                yaxis_title='Proportion of Team Salary Cap',
                yaxis=dict(range=[0.0, 1.1])
            )

            st.plotly_chart(overall_season_roster_status_cap_hit_prop_boxplot, use_container_width=True)

            st.write("""
            - Variability has increased over time (increasing size of boxplot boxes and size of boxplot whiskers)
            - Proportion of salary cap devoted to the active roster has trended down as proportion of salary cap devoted to the inactive roster has trended up
            """)

            st.write('---')

            tab4_team_list = sorted(spotrac_nfl_team_season_roster_df['team'].dropna().unique())
            selected_team = st.selectbox('Select Team', tab4_team_list)
            tab4_df = spotrac_nfl_team_season_roster_df[spotrac_nfl_team_season_roster_df['team'] == selected_team]

            # Initialize figure
            overall_season_roster_status_cap_hit_prop_team_point_boxplot = go.Figure()
            # Boxplot for all teams
            for status in spotrac_nfl_team_season_roster_df['roster_status'].unique():
                filtered = spotrac_nfl_team_season_roster_df[
                    spotrac_nfl_team_season_roster_df['roster_status'] == status
                    ]
                overall_season_roster_status_cap_hit_prop_team_point_boxplot.add_trace(go.Box(
                    x=filtered['season'],
                    y=filtered['cap_hit_prop'],
                    name=f'{status} (All Teams)',
                    boxpoints='outliers',
                    # marker_color='lightgray',
                    line=dict(width=1),
                    opacity=0.5
                ))
            # Overlay scatter for selected team
            overall_season_roster_status_cap_hit_prop_team_point_boxplot.add_trace(go.Scatter(
                x=tab4_df['season'],
                y=tab4_df['cap_hit_prop'],
                mode='markers',
                name=f'{selected_team} (Data Points)',
                marker=dict(
                    size=7,
                    color='red',
                    symbol='circle'
                ),
                text=tab4_df['roster_status'],
                hovertemplate='<b>%{text}</b><br>Season: %{x}<br>Cap %: %{y:.2%}<extra></extra>'
            ))
            # Layout adjustments
            overall_season_roster_status_cap_hit_prop_team_point_boxplot.update_layout(
                title=f"Proportion of Salary Cap by Roster Status per Season — {selected_team}",
                xaxis_title='NFL Season',
                yaxis_title='Proportion of Team Salary Cap',
                yaxis=dict(range=[0.0, 1.1]),
                legend=dict(title='Legend', x=1.05, y=1),
                margin=dict(r=200),
                height=600
            )
            st.plotly_chart(overall_season_roster_status_cap_hit_prop_team_point_boxplot, use_container_width=True)

            # Plotly lineplot
            overall_season_roster_status_cap_hit_prop_lineplot = px.line(
                tab4_df,
                x='season',
                y='cap_hit_prop',
                color='roster_status',
                title=f'{selected_team} — Proportion of Salary Cap by Roster Status per Season',
                labels={
                    'season': 'NFL Season',
                    'cap_hit_prop': 'Proportion of Team Salary Cap',
                    'roster_status': 'Roster Status'
                }
            )

            # Move legend outside of lineplot
            overall_season_roster_status_cap_hit_prop_lineplot.update_layout(
                legend=dict(
                    title='Roster Status',
                    x=1.05,
                    y=1
                ),
                margin=dict(r=150),  # Create space for legend
                xaxis_title='NFL Season',
                yaxis_title='Proportion of Team Salary Cap',
                yaxis=dict(range=[0.0, 1.1])
            )

            st.plotly_chart(overall_season_roster_status_cap_hit_prop_lineplot, use_container_width=True)

        # Optional: show selected data below
        with st.expander("View selected team data"):
            st.dataframe(tab4_df)

        with st.expander("Cap Hit Salary Proportion with Winning Pct Plots"):
            overall_season_roster_status_cap_hit_prop_winning_pct_active_df = spotrac_nfl_team_season_roster_df.loc[spotrac_nfl_team_season_roster_df['roster_status'] == 'active', :].copy()
            overall_season_roster_status_cap_hit_prop_winning_pct_active_df['season_str'] = overall_season_roster_status_cap_hit_prop_winning_pct_active_df['season'].astype(str)
            overall_season_roster_status_cap_hit_prop_winning_pct_inactive_df = spotrac_nfl_team_season_roster_df.loc[
                                                                              spotrac_nfl_team_season_roster_df[
                                                                                  'roster_status'] == 'inactive', :].copy()
            overall_season_roster_status_cap_hit_prop_winning_pct_inactive_df['season_str'] = \
            overall_season_roster_status_cap_hit_prop_winning_pct_inactive_df['season'].astype(str)


            overall_season_roster_status_cap_hit_prop_active_winning_pct_boxplot = px.box(
                overall_season_roster_status_cap_hit_prop_winning_pct_active_df,
                x='season_str',
                y='cap_hit_prop',
                title="Active Proportion of Salary Cap by Winning Pct per Season",
                labels={
                    'season': 'NFL Season',
                    'cap_hit_prop': 'Proportion of Team Salary Cap',
                    'winning_pct': 'Winning Pct'
                },
            )

            overall_season_roster_status_cap_hit_prop_active_winning_pct_boxplot.update_traces(
                line=dict(color='black', width=1),
                fillcolor='rgba(200, 200, 200, 0.2)',
                selector=dict(type='box')
            )

            overall_season_roster_status_cap_hit_prop_winning_pct_active_df_jitter = np.random.uniform(-0.25, 0.25, size = len(overall_season_roster_status_cap_hit_prop_winning_pct_active_df))
            overall_season_roster_status_cap_hit_prop_winning_pct_active_df['season_jittered'] = overall_season_roster_status_cap_hit_prop_winning_pct_active_df['season'] + overall_season_roster_status_cap_hit_prop_winning_pct_active_df_jitter

            overall_season_roster_status_cap_hit_prop_active_winning_pct_boxplot.add_trace(go.Scatter(
                x=overall_season_roster_status_cap_hit_prop_winning_pct_active_df['season_jittered'],
                y=overall_season_roster_status_cap_hit_prop_winning_pct_active_df['cap_hit_prop'],
                mode='markers',
                marker=dict(
                    size=8,
                    color=overall_season_roster_status_cap_hit_prop_winning_pct_active_df['pct'],
                    colorscale=[[0.0, 'blue'], [0.5, 'white'], [1.0, 'red']],
                    colorbar=dict(title='Win %'),
                    opacity=0.7
                ),
                text=overall_season_roster_status_cap_hit_prop_winning_pct_active_df['team'],
                hovertemplate="<b>%{text}</b><br>Season: %{x:.0f}<br>Cap %: %{y:.2f}<br>Win %: %{marker.color:.2f}<extra></extra>",
                showlegend=False
            ))

            # Layout tweaks
            overall_season_roster_status_cap_hit_prop_active_winning_pct_boxplot.update_layout(
                yaxis_title='Proportion of Team Salary Cap',
                xaxis_title='NFL Season',
                height=650,
                margin=dict(r=120)
            )

            st.plotly_chart(overall_season_roster_status_cap_hit_prop_active_winning_pct_boxplot, use_container_width=True)

            overall_season_roster_status_cap_hit_prop_inactive_winning_pct_boxplot = px.box(
                overall_season_roster_status_cap_hit_prop_winning_pct_inactive_df,
                x='season_str',
                y='cap_hit_prop',
                title="Inactive Proportion of Salary Cap by Winning Pct per Season",
                labels={
                    'season': 'NFL Season',
                    'cap_hit_prop': 'Proportion of Team Salary Cap',
                    'winning_pct': 'Winning Pct'
                },
            )

            overall_season_roster_status_cap_hit_prop_inactive_winning_pct_boxplot.update_traces(
                line=dict(color='black', width=1),
                fillcolor='rgba(200, 200, 200, 0.2)',
                selector=dict(type='box')
            )

            overall_season_roster_status_cap_hit_prop_winning_pct_inactive_df_jitter = np.random.uniform(-0.25, 0.25,
                                                                                                       size=len(
                                                                                                           overall_season_roster_status_cap_hit_prop_winning_pct_inactive_df))
            overall_season_roster_status_cap_hit_prop_winning_pct_inactive_df['season_jittered'] = \
            overall_season_roster_status_cap_hit_prop_winning_pct_inactive_df[
                'season'] + overall_season_roster_status_cap_hit_prop_winning_pct_inactive_df_jitter

            overall_season_roster_status_cap_hit_prop_inactive_winning_pct_boxplot.add_trace(go.Scatter(
                x=overall_season_roster_status_cap_hit_prop_winning_pct_inactive_df['season_jittered'],
                y=overall_season_roster_status_cap_hit_prop_winning_pct_inactive_df['cap_hit_prop'],
                mode='markers',
                marker=dict(
                    size=8,
                    color=overall_season_roster_status_cap_hit_prop_winning_pct_inactive_df['pct'],
                    colorscale=[[0.0, 'blue'], [0.5, 'white'], [1.0, 'red']],
                    colorbar=dict(title='Win %'),
                    opacity=0.7
                ),
                text=overall_season_roster_status_cap_hit_prop_winning_pct_inactive_df['team'],
                hovertemplate="<b>%{text}</b><br>Season: %{x:.0f}<br>Cap %: %{y:.2f}<br>Win %: %{marker.color:.2f}<extra></extra>",
                showlegend=False
            ))

            # Layout tweaks
            overall_season_roster_status_cap_hit_prop_inactive_winning_pct_boxplot.update_layout(
                yaxis_title='Proportion of Team Salary Cap',
                xaxis_title='NFL Season',
                height=650,
                margin=dict(r=120)
            )

            st.plotly_chart(overall_season_roster_status_cap_hit_prop_inactive_winning_pct_boxplot,
                            use_container_width=True)

            st.write("""
            Of the Active Roster and Inactive Roster portions of the Salary Cap, the following observations are made:
            - Active proportions and inactive proportions of the salary cap are slowly trending towards each other
                - More salary cap space is devoted to the inactive roster over time
                    - Changes to CBA?
                    - Changes to team approach toward players that can't be on the active roster due to injury or other reasons?
                    - Changes to practice squad sizes?
                    - Teams needing to change practice and recovery strategies to improve recovery to minimize injuries that result in inactive players? 
            - Greater than .500 Winning % tends to be in upper half of Active Salary Cap Proportion Boxplots
                - Better performing teams devote most of their salary cap to players on the active roster
                - Better performing teams have "better luck, better recovery, better training, etc." that leads to few players (less salary cap) on the inactive roster
            - Less than .500 Winning % tends to be in lower half of Active Salary Cap Proportion Boxplots
                - Worse performing teams devote more of their salary cap to players on the inactive roster than the active roster
                - Worse performing teams have "worse luck, worse recovery, worse training, etc." that leads to more players (more salary cap) on the inactive roster
            - 2011 - 2016:
                - Teams (colored points), by visual inspection, appear to show less divergence (dispersion)
                - Better performing teams (blue) and worse performing teams (red) have greater mixing
            - 2017 - 2024:
                - Teams (colored points), by visual inspection, appear to show greater divergence (dispersion)
                - Better performing teams (blue) and worse performing teams (red) have less mixing
            """)

    with tab5:
        st.markdown("#### Team - Season - Roster Status Unsupervised Learning")
        st.write("""
        What underlying structures are there to the Spotrac salary cap data?
        """)
        with st.expander("Methodology"):
            st.write("""
            - Dataset for analysis was the spotrac_nfl_team_season_roster_wide_df
            - Dataframe is in wide format, so each team-season combination is an observation and each column describes the team-season combination
                - `player_count_prop` and `cap_hit_prop` fields for active and inactive roster statuses add up to 1.0, so only the active roster status columns are used
            - The resulting dataset used for unsupervised learning contains 448 observations and 6 columns
                - `player_count_prop_active`
                - `cap_hit_prop_active`
                - `cap_hit_per_player_prop_active`
                - `ratio_cap_hit_per_player_prop`
                - `ratio_cap_hit_prop`
                - `ratio_player_count_prop`
            - Three different unsupervised learning models were used:
                - KMeans clustering
                - Gaussian Mixture Model
                - DBSCAN clustering
            - After performing unsupervised learning, cluster assignments for each row are applied to the original dataset, spotrac_nfl_team_season_roster_wide_df
            - The original dataset with cluster assignments is then grouped by cluster and the mean values for each cluster are calculated 
            """)

        with st.expander("Original and Filtered Dataset used for clustering"):
            st.write("Original Dataset: spotrac_nfl_team_season_roster_wide_df")
            st.dataframe(spotrac_nfl_team_season_roster_wide_df)
            st.write("---")
            st.write("Clustering Dataset: Filtered spotrac_nfl_team_season_roster_df")
            st.dataframe(spotrac_nfl_team_season_roster_wide_df.loc[:, ['cap_hit_prop_active',
                                                                        'cap_hit_per_player_prop_active',
                                                                          'player_count_prop_active',
                                                                          'ratio_cap_hit_per_player_prop',
                                                                          'ratio_cap_hit_prop',
                                                                          'ratio_player_count_prop']])

        with st.expander("KMeans Clustering"):
            st.write("""
            - Elbow Plot and Average Cluster Silhouette Score plot along with PC1 and PC2 cluster plot
            """)
            elbow_plot_col, silhouette_score_plot_col, kmc_pc12_plot_col = st.columns(3)
            with elbow_plot_col:
                kmc_elbow_plot_path = project_data_exports_path/'kmc_elbow_plot.png'
                st.image(
                    str(kmc_elbow_plot_path),
                    caption='KMC Elbow Plot',
                    use_container_width=True
                )
            with silhouette_score_plot_col:
                silhouette_score_plot_path = project_data_exports_path/'kmc_silhouette_score_plot.png'
                st.image(
                    str(silhouette_score_plot_path),
                    caption='KMC Silhouette Score Plot',
                    use_container_width=True
                )
            with kmc_pc12_plot_col:
                kmc_pc12_plot_path = project_data_exports_path/'kmc_pc12_cluster_plot.png'
                st.image(
                    str(kmc_pc12_plot_path),
                    caption='Cluster Plot Projection of PC1 and PC2',
                    use_container_width=True
                )
            st.write('---')
            st.write('spotrac_nfl_team_season_roster_wide_df with KMeans Cluster Assignments')
            kmc_labeled_df_clusters = kmc_labeled_df['label'].unique()
            kmc_labeled_df_clusters_choice = st.multiselect('Select KMeans Cluster(s) to View', kmc_labeled_df_clusters, default=kmc_labeled_df_clusters)
            st.dataframe(kmc_labeled_df.loc[kmc_labeled_df['label'].isin(kmc_labeled_df_clusters_choice), :])
            st.write("---")
            kmc_labeled_df_numerical_cols = ['season', 'player_count_active', 'player_count_inactive',
                                             'cap_hit_sum_active', 'cap_hit_sum_inactive',
                                             'player_count_prop_active', 'player_count_prop_inactive',
                                             'w', 'l', 'pct', 'pf', 'pa', 'net_pts',
                                             'div_win_pct', 'conf_win_pct', 'pc_1', 'pc_2',]
            kmc_labeled_df_categorical_cols = ['team', 'label']
            kmc_labeled_df_color_cols = kmc_labeled_df_numerical_cols + kmc_labeled_df_categorical_cols
            kmc_col1, kmc_col2, kmc_col3 = st.columns(3)
            with kmc_col1:
                kmc_x_col = st.selectbox('Select X-Axis Column', options=kmc_labeled_df_numerical_cols,
                                         index=kmc_labeled_df_numerical_cols.index('pc_1'), key='kmc_x_axis')
            with kmc_col2:
                kmc_y_col = st.selectbox('Select Y-Axis Column', options=kmc_labeled_df_numerical_cols,
                                         index=kmc_labeled_df_numerical_cols.index('pc_2'), key='kmc_y_axis')
            with kmc_col3:
                kmc_color_col = st.selectbox('Select Color', options=kmc_labeled_df_color_cols,
                                             index=kmc_labeled_df_color_cols.index('label'), key='kmc_color')

            if not kmc_labeled_df.empty:
                is_discrete = kmc_color_col in kmc_labeled_df_categorical_cols or (kmc_color_col == 'label') or kmc_labeled_df[kmc_color_col].nunique() < 10
                kmc_color_param = dict(
                    color=kmc_color_col,
                    color_discrete_sequence=px.colors.qualitative.Plotly if is_discrete else None,
                    color_continuous_scale=None if is_discrete else 'Viridis'
                )

                kmc_labeled_df_scatterplot = px.scatter(
                    kmc_labeled_df,
                    x=kmc_x_col,
                    y=kmc_y_col,
                    **kmc_color_param,
                    hover_data=['team', 'season'],
                    opacity=0.7,
                    size_max=10
                )

                # Hide color bar for discrete colors
                if is_discrete:
                    kmc_labeled_df_scatterplot.update_traces(marker=dict(showscale=False))

                # Update layout
                kmc_labeled_df_scatterplot.update_layout(
                    title=f"Scatter Plot: {kmc_x_col} vs {kmc_y_col} (Colored by {kmc_color_col})",
                    xaxis_title=kmc_x_col.replace('_', ' ').title(),
                    yaxis_title=kmc_y_col.replace('_', ' ').title(),
                    legend_title=kmc_color_col.replace('_', ' ').title(),
                    template='plotly_white',
                    height=600,
                    showlegend=True
                )

                st.plotly_chart(kmc_labeled_df_scatterplot, use_container_width=True)

            st.write('---')
            st.write('Cluster means for spotrac_nfl_team_season_roster_wide_df')
            st.dataframe(kmc_grouped_clusters_labeled_df)
            st.write("---")
            st.write('Cluster team-season values for spotrac_nfl_team_season_roster_wide_df')
            st.dataframe(kmc_grouped_clusters_team_labeled_df)
            st.write("""
            ## Observations on KMeans Cluster Groupings

            These insights are drawn from the average values of each KMeans cluster across key salary cap, roster, and performance metrics.

            ---

            ### Cluster 0 (n = 151)
            - **Top-performing cluster** across the board:
              - **Win percentage:** 61.4%
              - **Net points:** +55
              - **Avg. points scored:** 407
            - **Cap strategy:** 
              - ~82% of salary cap allocated to **active roster**
              - Only **~40% of players** with a cap hit are on the active roster
              - **Highest cap hit per active player** (~2.08x)
            - **Representative team-seasons:**
              - **2015 Carolina Panthers** – 15-1, +192 net points
              - **2024 Kansas City Chiefs** – 15-2, +59 net points
              - **2020 Kansas City Chiefs** – 14-2, +111 net points
            - Interpretation: Small, high-cap-value core players = efficient and successful

            ---

            ### Cluster 1 (n = 20)
            - **Roster-heavy active strategy**:
              - ~80% of players with a cap hit are active (highest among clusters)
              - **Win %:** 49.7% — **league average**
            - **Cap efficiency:** Highest `cap_hit_prop_active` (~95%) but **lowest cap hit per player**
            - **Representative team-seasons:**
              - **2011 Green Bay Packers** – 15-1, +201 net points
              - **2011 New Orleans Saints** – 13-3, +208 net points
              - **2011 San Francisco 49ers** – 13-3, +151 net points
            - Interpretation: Teams invested in continuity and participation, but **results were average**

            ---

            ### Cluster 2 (n = 47)
            - Similar to Cluster 1 but with:
              - Fewer active players (~63%)
              - Slightly lower `cap_hit_prop_active` (~80%)
            - **Win %:** 50.4% — marginally above average
            - **Offense:** ~368 points scored (moderate)
            - **Representative team-seasons:**
              - **2012 Atlanta Falcons** – 13-3, +120 net points
              - **2012 Denver Broncos** – 13-3, +192 net points
              - **2013 Denver Broncos** – 13-3, +207 net points
            - Interpretation: Balanced approach, but didn’t generate big performance gains

            ---

            ### Cluster 3 (n = 230)
            - **Lowest-performing group**:
              - **Win %:** 42.5%
              - **Net points:** -36
            - **Cap strategy:** 
              - Only ~64% of cap used on active players
              - Active roster holds ~38% of cap-hit players
              - **Representative team-seasons:**
                - **2024 Detroit Lions** – 15-2, +222 net points
                - **2019 Baltimore Ravens** – 14-2, +249 net points
                - **2024 Minnesota Vikings** – 14-3, +100 net points
            - Interpretation: High cost on inactive players may have hindered effectiveness
            - **Note:** Despite generally lower average performance, Cluster 3 contains **high-performing outliers** — likely driven by strong quarterback play or isolated high-value seasons that overcame inefficient roster structures.

            ---

            ### Summary:
            - **Cluster 0 = Elite Efficiency**: Small active core, high investment, top-tier results.
            - **Cluster 1 = Full Participation**: Most players active, cap fully used, average outcomes.
            - **Cluster 2 = Balanced Strategy**: Slightly better results than Cluster 1, but fewer players active.
            - **Cluster 3 = Inefficiency**: High inactive cap cost, low performance metrics.

            """)
        
        with st.expander("Gaussian Mixture Model (GMM) Clustering"):
            st.write("""
            - BIC and AIC vs Number of Components curves indicate 4 clusters as optimal cluster quantity
            """)
            gmm_bic_aic_plot_col, gmm_pc12_plot_col = st.columns(2)
            with gmm_bic_aic_plot_col:
                gmm_bic_aic_plot_path = project_data_exports_path / 'gmm_bic_aic_plot.png'
                st.image(
                    str(gmm_bic_aic_plot_path),
                    caption='GMM BIC and AIC vs Number of Components Plot',
                    use_container_width=True
                )
            with gmm_pc12_plot_col:
                gmm_pc12_plot_path = project_data_exports_path / 'gmm_pc12_cluster_plot.png'
                st.image(
                    str(gmm_pc12_plot_path),
                    caption='Cluster Plot Projection of PC1 and PC2',
                    use_container_width=True
                )

            st.write('spotrac_nfl_team_season_roster_df with GMM Cluster Assignments')
            gmm_labeled_df_clusters = gmm_labeled_df['label'].unique()
            gmm_labeled_df_clusters_choice = st.multiselect('Select GMM Cluster(s) to View', gmm_labeled_df_clusters,
                                                            default=gmm_labeled_df_clusters)
            st.dataframe(gmm_labeled_df.loc[gmm_labeled_df['label'].isin(gmm_labeled_df_clusters_choice), :])
            st.write("---")

            gmm_labeled_df_numerical_cols = ['season', 'player_count_active', 'player_count_inactive',
                                             'cap_hit_sum_active', 'cap_hit_sum_inactive',
                                             'player_count_prop_active', 'player_count_prop_inactive',
                                             'w', 'l', 'pct', 'pf', 'pa', 'net_pts',
                                             'div_win_pct', 'conf_win_pct', 'pc_1', 'pc_2',]
            gmm_labeled_df_categorical_cols = ['team', 'label']
            gmm_labeled_df_color_cols = gmm_labeled_df_numerical_cols + gmm_labeled_df_categorical_cols
            gmm_col1, gmm_col2, gmm_col3 = st.columns(3)
            with gmm_col1:
                gmm_x_col = st.selectbox('Select X-Axis Column', options=gmm_labeled_df_numerical_cols,
                                         index=gmm_labeled_df_numerical_cols.index('pc_1'), key='gmm_x_axis')
            with gmm_col2:
                gmm_y_col = st.selectbox('Select Y-Axis Column', options=gmm_labeled_df_numerical_cols,
                                         index=gmm_labeled_df_numerical_cols.index('pc_2'), key='gmm_y_axis')
            with gmm_col3:
                gmm_color_col = st.selectbox('Select Color', options=gmm_labeled_df_color_cols,
                                             index=gmm_labeled_df_color_cols.index('label'), key='gmm_color')

            if not gmm_labeled_df.empty:
                is_discrete = gmm_color_col in gmm_labeled_df_categorical_cols or (gmm_color_col == 'label') or \
                              gmm_labeled_df[gmm_color_col].nunique() < 10
                gmm_color_param = dict(
                    color=gmm_color_col,
                    color_discrete_sequence=px.colors.qualitative.Plotly if is_discrete else None,
                    color_continuous_scale=None if is_discrete else 'Viridis'
                )

                gmm_labeled_df_scatterplot = px.scatter(
                    gmm_labeled_df,
                    x=gmm_x_col,
                    y=gmm_y_col,
                    **gmm_color_param,
                    hover_data=['team', 'season'],
                    opacity=0.7,
                    size_max=10
                )

                # Hide color bar for discrete colors
                if is_discrete:
                    gmm_labeled_df_scatterplot.update_traces(marker=dict(showscale=False))

                # Update layout
                gmm_labeled_df_scatterplot.update_layout(
                    title=f"Scatter Plot: {gmm_x_col} vs {gmm_y_col} (Colored by {gmm_color_col})",
                    xaxis_title=gmm_x_col.replace('_', ' ').title(),
                    yaxis_title=gmm_y_col.replace('_', ' ').title(),
                    legend_title=gmm_color_col.replace('_', ' ').title(),
                    template='plotly_white',
                    height=600,
                    showlegend=True
                )

                st.plotly_chart(gmm_labeled_df_scatterplot, use_container_width=True)

            st.write('Cluster means for spotrac_nfl_team_season_roster_df')
            st.dataframe(gmm_grouped_clusters_labeled_df)
            st.write("---")
            st.write('Cluster team-season values for spotrac_nfl_team_season_roster_df')
            st.dataframe(gmm_grouped_clusters_team_labeled_df)
            st.write("""
            ## Observations on GMM Cluster Groupings

            This analysis summarizes key differences in salary cap structure, roster strategy, and performance across GMM-identified clusters.

            ---

            ### Cluster 4 (n = 48)
            - **Top performance cluster:**
              - **Win %:** 66.5%
              - **Net points:** +78
              - **Points scored:** 422 (highest)
            - **Cap strategy:**
              - 88% of cap spent on active roster
              - 43% of cap-hit players on active roster
              - **Highest cap hit per active player**
            - **Representative team-seasons:**
              - **2024 Kansas City Chiefs** – 15-2, +59 net points
              - **2020 Kansas City Chiefs** – 14-2, +111 net points
              - **2022 Kansas City Chiefs** – 14-3, +127 net points
            - Interpretation: Small, high-value active core with elite results
            - *Note: Although this cluster had elite results, its relatively small size (n = 48) may reflect a specialized strategy rather than a league-wide trend.*

            ---

            ### Cluster 0 (n = 206)
            - **Strong performer:**
              - **Win %:** 54.6%
              - **Net points:** +24
            - **Cap efficiency:**
              - 75% of cap on active players
              - 39% of cap-hit players active
            - **Representative team-seasons:**
              - **2015 Carolina Panthers** – 15-1, +192 net points
              - **2024 Detroit Lions** – 15-2, +222 net points
              - **2019 Baltimore Ravens** – 14-2, +249 net points
            - Interpretation: Lean rosters with strategic investment in top-tier players

            ---

            ### Cluster 2 (n = 54)
            - **Mid-tier performance:**
              - **Win %:** 48.7%
              - **Net points:** -7
            - **Roster strategy:**
              - 78% of cap spent on active roster
              - 62% of cap-hit players are active
            - **Representative team-seasons:**
              - **2012 Atlanta Falcons** – 13-3, +120 net points
              - **2012 Denver Broncos** – 13-3, +192 net points
              - **2013 Denver Broncos** – 13-3, +207 net points
            - Interpretation: High continuity but middling performance

            ---

            ### Cluster 1 (n = 18)
            - **Similar to Cluster 2** but with:
              - Even **higher active player share (81%)**
              - **Highest cap % on active players** (95%)
              - **Performance:** Win % = 49.0%, Net pts = -8.9
            - **Representative team-seasons:**
              - **2011 Green Bay Packers** – 15-1, +201 net points
              - **2011 New Orleans Saints** – 13-3, +208 net points
              - **2011 San Francisco 49ers** – 13-3, +151 net points
            - Interpretation: Full-roster utilization doesn’t guarantee success
            - *Note: This cluster includes only 18 team-seasons, so results should be interpreted with caution.*

            ---

            ### Cluster 3 (n = 122)
            - **Lowest performance:**
              - **Win %:** 36.5%
              - **Net points:** -66
              - **Points allowed:** 402 (highest)
            - **Cap structure:**
              - Only 58% of cap on active players
              - 35% of players on the active roster
            - **Representative team-seasons:**
              - **2024 Minnesota Vikings** – 14-3, +100 net points
              - **2022 Philadelphia Eagles** – 14-3, +133 net points
              - **2014 Dallas Cowboys** – 12-4, +115 net points
            - Interpretation: High inactive cap burden → poor results

            ---

            ### Summary:
            - **Cluster 4:** Elite performance from tight, expensive cores
            - **Cluster 0:** Efficient, lean rosters = consistent winning
            - **Clusters 1 & 2:** Broad participation but only average outcomes
            - **Cluster 3:** Inefficiency leads to underperformance

            """)

        with st.expander("DBSCAN Clustering"):
            st.write("""
            - Sorted 5th Nearest Neighbor Distances plot and Silhouette Score Plot
            """)
            dbscan_distance_plot_col, dbscan_silhouette_score_plot_col, dbscan_pc12_plot_col = st.columns(3)
            with dbscan_distance_plot_col:
                dbscan_distance_plot_path = project_data_exports_path / 'dbscan_density_fig.png'
                st.image(
                    str(dbscan_distance_plot_path),
                    caption='DBSCAN 5th Nearest Neighbor Distances Plot',
                    use_container_width=True
                )
            with dbscan_silhouette_score_plot_col:
                dbscan_silhouette_score_plot_path = project_data_exports_path / 'dbscan_silhouette_score_plot.png'
                st.image(
                    str(dbscan_silhouette_score_plot_path),
                    caption='DBSCAN Silhouette Score Plot',
                    use_container_width=True
                )
            with dbscan_pc12_plot_col:
                dbscan_pc12_plot_path = project_data_exports_path / 'dbscan_pc12_cluster_plot.png'
                st.image(
                    str(dbscan_pc12_plot_path),
                    caption='Cluster Plot Projection of PC1 and PC2',
                    use_container_width=True
                )
                
            st.write('---')
            st.write('spotrac_nfl_team_season_roster_wide_df with DBSCAN Cluster Assignments')
            dbscan_labeled_df_clusters = dbscan_labeled_df['label'].unique()
            dbscan_labeled_df_clusters_choice = st.multiselect('Select KMeans Cluster(s) to View', dbscan_labeled_df_clusters,
                                                            default=dbscan_labeled_df_clusters)
            st.dataframe(dbscan_labeled_df.loc[dbscan_labeled_df['label'].isin(dbscan_labeled_df_clusters_choice), :])
            st.write("---")
            dbscan_labeled_df_numerical_cols = ['season', 'player_count_active', 'player_count_inactive',
                                             'cap_hit_sum_active', 'cap_hit_sum_inactive',
                                             'player_count_prop_active', 'player_count_prop_inactive',
                                             'w', 'l', 'pct', 'pf', 'pa', 'net_pts',
                                             'div_win_pct', 'conf_win_pct', 'pc_1', 'pc_2',]
            dbscan_labeled_df_categorical_cols = ['team', 'label']
            dbscan_labeled_df_color_cols = dbscan_labeled_df_numerical_cols + dbscan_labeled_df_categorical_cols
            dbscan_col1, dbscan_col2, dbscan_col3 = st.columns(3)
            with dbscan_col1:
                dbscan_x_col = st.selectbox('Select X-Axis Column', options=dbscan_labeled_df_numerical_cols,
                                         index=dbscan_labeled_df_numerical_cols.index('pc_1'), key='dbscan_x_axis')
            with dbscan_col2:
                dbscan_y_col = st.selectbox('Select Y-Axis Column', options=dbscan_labeled_df_numerical_cols,
                                         index=dbscan_labeled_df_numerical_cols.index('pc_2'), key='dbscan_y_axis')
            with dbscan_col3:
                dbscan_color_col = st.selectbox('Select Color', options=dbscan_labeled_df_color_cols,
                                             index=dbscan_labeled_df_color_cols.index('label'), key='dbscan_color')
            if not dbscan_labeled_df.empty:
                is_discrete = dbscan_color_col in dbscan_labeled_df_categorical_cols or (dbscan_color_col == 'label') or dbscan_labeled_df[dbscan_color_col].nunique() < 10
                dbscan_color_param = dict(
                    color=dbscan_color_col,
                    color_discrete_sequence=px.colors.qualitative.Plotly if is_discrete else None,
                    color_continuous_scale=None if is_discrete else 'Viridis'
                )

                dbscan_labeled_df_scatterplot = px.scatter(
                    dbscan_labeled_df,
                    x=dbscan_x_col,
                    y=dbscan_y_col,
                    **dbscan_color_param,
                    hover_data=['team', 'season'],
                    opacity=0.7,
                    size_max=10
                )

                # Hide color bar for discrete colors
                if is_discrete:
                    dbscan_labeled_df_scatterplot.update_traces(marker=dict(showscale=False))

                # Update layout
                dbscan_labeled_df_scatterplot.update_layout(
                    title=f"Scatter Plot: {dbscan_x_col} vs {dbscan_y_col} (Colored by {dbscan_color_col})",
                    xaxis_title=dbscan_x_col.replace('_', ' ').title(),
                    yaxis_title=dbscan_y_col.replace('_', ' ').title(),
                    legend_title=dbscan_color_col.replace('_', ' ').title(),
                    template='plotly_white',
                    height=600,
                    showlegend=True
                )

                st.plotly_chart(dbscan_labeled_df_scatterplot, use_container_width=True)
            st.write('Cluster means for spotrac_nfl_team_season_roster_wide_df')
            st.dataframe(dbscan_grouped_clusters_labeled_df)
            st.write("---")
            st.write('Cluster team-season values for spotrac_nfl_team_season_roster_wide_df')
            st.dataframe(dbscan_grouped_clusters_team_labeled_df)
            st.write("""
            ## Observations on DBSCAN Cluster Groupings

            This analysis summarizes cap structure, roster allocation, and performance outcomes for the two clusters discovered by DBSCAN.

            ---

            ### Cluster -1 (Noise Cluster, n = 12)
            - **Roster & Cap:**
              - 73% of cap-hit players are active
              - **95% of cap** is spent on the active roster (highest of both clusters)
            - **Performance:**
              - **Win %:** 60.7% (strongest performance)
              - **Net points:** +37
              - **Points scored:** ~400, allowed: ~362
            - **Representative team-seasons:**
              - **2011 Green Bay Packers**: 15-1, +201 net points
              - **2011 New England Patriots**: 13-3, +171 net points
              - **2020 Buffalo Bills**: 13-3, +126 net points
            - **Interpretation:** Despite being identified as "noise," these teams had **lean, highly efficient rosters** and strong results. Possibly elite or outlier teams with unique roster strategies.
            - *Note: This cluster was flagged as noise by DBSCAN, meaning these team-seasons did not conform to any broader grouping. They may represent **elite outliers** rather than repeatable strategies.*
            ---

            ### Cluster 0 (n = 436)
            - **Roster & Cap:**
              - 42% of cap-hit players are active
              - ~73% of the cap is spent on them
            - **Performance:**
              - **Win %:** 49.7% (league average)
              - **Net points:** -1
              - **Points scored:** ~370, allowed: ~371
            - **Representative team-seasons:**
              - **2015 Carolina Panthers**: 15-1, +192 net points
              - **2024 Detroit Lions**: 15-2, +222 net points
              - **2024 Kansas City Chiefs**: 15-2, +59 net points
            - **Interpretation:** Large, rotational rosters with **average investment efficiency and performance**. Reflects typical league behavior.

            ---

            ### Summary
            - **Cluster -1** (Noise): These teams stand out for exceptional **efficiency and scoring**. Possibly elite rosters with minimal waste.
            - **Cluster 0**: Most teams fall here — **middle-of-the-road** outcomes and standard cap/roster distributions.

            """)

        with st. expander("Clustering Takeaways"):
            st.write("""
            ## Generalized Clustering Takeaways

            This summary synthesizes trends across KMeans, GMM, and DBSCAN clustering results, offering high-level strategic insights about roster construction and salary cap efficiency.

            ---

            ### Cap Efficiency > Roster Size
            - Across all models, clusters with **high `cap_hit_prop_active` (80–95%)** and **lower `player_count_prop_active` (35–45%)** consistently:
              - Win more games
              - Score more points
              - Post higher net point differentials
            - These teams tend to concentrate cap space in a **smaller core of impactful players**.
            - **Examples:**
              - KMeans Cluster 0: *2015 Cardinals*, *2021 Cardinals*, *2017 Falcons*
              - GMM Cluster 4: *2013 Broncos*, *2015 Cardinals*, *2016 Cowboys*
              - DBSCAN Cluster -1: *2011 Packers*, *2020 Bills*, *2011 Patriots*

            ---

            ### Deep Rosters Don’t Guarantee Success
            - Clusters with **very high `player_count_prop_active` (~0.75–0.80)** and full cap utilization showed:
              - **Average win percentages (~49–50%)**
              - Low-to-moderate net point gains or deficits
            - Participation ≠ performance. These teams spread cap across more players but didn’t dominate.
            - Examples:
              - KMeans Cluster 1: *TBD team-seasons*
              - GMM Cluster 1: *2019 Buccaneers*, *2018 Vikings*
              - GMM Cluster 2: *2020 Washington*, *2015 Jets*

            ---

            ### Inefficiency = Poor Outcomes
            - Clusters with **low `cap_hit_prop_active` (≤65%)** and **high inactive burden** underperformed:
              - Win % drops below 45%
              - Net points typically negative
            - These teams paid many inactive players—**a drag on performance**.
            - Examples:
              - KMeans Cluster 3: *2019 Dolphins*, *2021 Giants*
              - GMM Cluster 3: *2014 Raiders*, *2015 Browns*

            ---

            ### Model-Consistent Patterns
            - **Elite clusters** (e.g., KMeans 0, GMM 4, DBSCAN -1) **share characteristics**:
              - Small active cores
              - High cap investment efficiency
              - Strong point differentials
            - **Average-performing clusters** (e.g., KMeans 1/2, GMM 1/2, DBSCAN 0) tend toward:
              - Larger active groups
              - Full cap deployment
              - Moderate scoring and win %

            ---

            ### Caveats & Special Cases
            - **GMM Cluster 1** (n = 18) and **Cluster 4** (n = 48) are relatively small.
              - Results may reflect **unique or non-representative strategies**.
            - **DBSCAN Cluster -1** (n = 12) was **labeled noise**, meaning teams were outliers:
              - Despite strong performance, this cluster isn’t a trend—it may capture **elite exceptions**.
            - **Interpret with care** when generalizing from small, high-performing groups.

            ---

            ### Bottom Line:
            > Elite teams consistently concentrate cap space in a **smaller number of high-value active players**.  
            > Larger rosters with full participation **do not necessarily produce better outcomes**, while inefficient rosters with heavy inactive costs tend to underperform.

            """)

    with tab6:
        st.markdown("#### Team - Season - Roster Status Supervised Learning")
        st.write("""
        Predicting a team's season winning percentage based on the proportion of its salary cap allocated to players 
        on its active roster and the proportion of its players on the active roster.
        """)
        with st.expander("Methodology"):
            st.write("""
            - Dataset for analysis was the spotrac_nfl_team_season_roster_wide_df
            - Dataframe is in wide format, so each team-season combination has feature pairs, one for each `roster_status` (active and inactive)
                - `player_count_prop` and `cap_hit_prop` fields for active and inactive roster statuses add up to 1.0, so only the active roster status are used
            - The resulting dataset used for supervised learning contained 448 observations and 3 columns (`pct`, 
            `player_count_prop_active`,  and `cap_hit_prop_active`)
                - `season` was not used as it would prevent out-of-sample predictions from being performed
            - The dataset was split into two subsets: 
                - X: the independent variables `cap_hit_prop_active` and `player_count_prop_active`
                - y: the dependent variable `pct`
            - Using scikit-learn's train_test_split, the X and y datasets were split into training and test 
            splits,  33% of the 448 observations went to the test dataset, and 67% of the observations went to the training dataset
            - 8 different regression algorithms were trained on the training dataset and then subsequently tested
                - Linear Regression
                - K-Nearest Neighbors Regression
                - Decision Tree Regression
                - Random Forest Regression
                - Ridge Regression
                - LASSO Regression
                - Elastic Net Regression
                - XGBoost Regression
            - Using scikit-learn's pipeline, each model's training and prediction workflow was standardized to ensure consistency and to prevent data leakage
            - The independent variables were not standardized for the Linear Regression and K-Nearest Neighbors models. These variables were on the same scale, from 0-1. The independent variables were standardized using scikit-learn's StandardScaler for the Ridge, LASSO, and ElasticNet models in order to aid in those algorithm's performance. 
            - The battery of models were trained because each model can provide different insights into the data and 
            taken together, could provide a better picture into the relationship between the independent and 
            dependent variables
                - For example, the Decision Tree Regression model provides a decision tree, Decision Tree, Random 
                Forest, and XGBoost provide feature importance information, and Ridge, LASSO, ElasticNet Regression models shrink 
                coefficients for those features that are less important for predicting the target variable.
            - When training the models, scikit-learn's GridSearchCV function was used to find the optimal hyperparameters
            - After generating predictions using the test set, plots were generated to ascertain the ability of the 
            models to predict the dependent variable, `pct`
            - All models were then tested on the full 448 observation dataset to provide a final assessment of model 
            performance.
            """)
        with st.expander('Original and Filtered Dataset used for regression model training'):
            st.write("Original Dataset: spotrac_nfl_team_season_roster_df")
            st.dataframe(spotrac_nfl_team_season_roster_wide_df)
            st.write('---')
            st.write('Regression model training dataset: Filtered spotrac_nfl_team_season_roster_df')
            st.dataframe(spotrac_nfl_team_season_roster_wide_df[['pct', 'player_count_prop_active', 'cap_hit_prop_active']])

        with st.expander('Model Preprocessing Summary'):
            st.markdown("#### Model Preprocessing Summary")

            models = {
                "Linear Regression": lr_model_pt_1,
                "Decision Tree": tree_model_pt_1,
                "K-Nearest Neighbors": knn_model_pt_1,
                "Random Forest": rf_model_pt_1,
                "Ridge Regression": ridge_model_pt_1,
                "Lasso Regression": lasso_model_pt_1,
                "ElasticNet": elasticnet_model_pt_1,
                "XGBoost": xgbr_model_pt_1
            }

            model_summary_df = summarize_model_preprocessors(models)
            st.dataframe(model_summary_df, use_container_width=True)

        with st.expander('View Regression Model Diagnostics'):
            tab6col1, tab6col2 = st.columns(2)
            with tab6col1:
                st.write("Model Training RMSE Table")
                st.dataframe(supervised_learning_pt_1_model_results_df)
                st.write('')
            with tab6col2:
                st.write("Model Training RMSE Plot")
                # Display the model_perf_rmse_plot.png
                rmse_plot_path = project_data_exports_path / 'model_perf_rmse_plot.png'
                if rmse_plot_path.exists():
                    st.image(
                        str(rmse_plot_path),
                        caption="Model Performance RMSE Plot",
                        use_container_width=True
                    )
                else:
                    st.error(
                        f"RMSE plot not found at {rmse_plot_path}. Please ensure the file 'model_perf_rmse_plot.png' exists in the specified directory."
                    )
            st.write('Linear Model Coefficients Table')
            st.dataframe(supervised_learning_pt_1_linear_model_coefs_df)

            feature_names_pt_1 = ['player_count_prop_active', 'cap_hit_prop_active']
            models = {
                "Linear Regression": lr_model_pt_1,
                "K-Nearest Neighbors": knn_model_pt_1,
                "Decision Tree": tree_model_pt_1,
                "Random Forest": rf_model_pt_1,
                "Ridge Regression": ridge_model_pt_1,
                "Lasso Regression": lasso_model_pt_1,
                "ElasticNet": elasticnet_model_pt_1,
                "XGBoost": xgbr_model_pt_1
            }

            model_name_mapping_pt_1 = {
                'Linear Regression': 'lr',
                "K-Nearest Neighbors": "knn",
                "Decision Tree": "tree",
                "Random Forest": "rf",
                "Ridge Regression": "ridge",
                "Lasso Regression": "lasso",
                "ElasticNet": "elasticnet",
                "XGBoost": "xgbr"
            }

            # Dropdown to select which model's diagnostics to view
            selected_regression_model_pt_1 = st.selectbox(
                "Select Model to View Diagnostics",
                options=list(models.keys()),
                index=0
            )

            # Get the shorthand name for the selected model
            regression_model_shorthand_pt_1 = model_name_mapping_pt_1[selected_regression_model_pt_1]

            # Construct the path to the PNG file
            regression_model_diagnostics_png_path_pt_1 = project_data_exports_path / f"{regression_model_shorthand_pt_1}_pred_error_residuals_qq_plot.png"

            # Check if the file exists and display it
            if regression_model_diagnostics_png_path_pt_1.exists():
                st.image(
                    str(regression_model_diagnostics_png_path_pt_1),
                    caption=f"Model Diagnostics for {selected_regression_model_pt_1}: Prediction Error, Residuals, and Q-Q Plot",
                    use_container_width=True
                )
            else:
                st.error(
                    f"Diagnostics plot for {selected_regression_model_pt_1} not found at {regression_model_diagnostics_png_path_pt_1}. Please ensure the PNG file has been generated.")

            original_feature_names_pt_1 = [
                'cap_hit_prop_active',
                'cap_hit_per_player_prop_active',
                'player_count_prop_active',
                'ratio_cap_hit_per_player_prop',
                'ratio_cap_hit_prop',
                'ratio_player_count_prop'
            ]

            # Extract model information on the fly
            try:
                info = extract_model_info(models[selected_regression_model_pt_1], regression_model_shorthand_pt_1, original_feature_names_pt_1)

                # Display coefficients for Linear Regression, Ridge, Lasso, ElasticNet
                if "coefficients" in info:
                    st.markdown(f"**Coefficients for {selected_regression_model_pt_1}**")
                    coef_df = pd.DataFrame.from_dict(info["coefficients"], orient='index', columns=['Coefficient'])
                    coef_df['Coefficient'] = coef_df['Coefficient'].apply(lambda x: f"{x:.4f}")
                    st.dataframe(coef_df, use_container_width=True)

                # Display decision tree structure for Decision Tree
                if "decision_tree" in info:
                    st.markdown(f"**Decision Tree Structure for {selected_regression_model_pt_1}**")
                    st.code(info["decision_tree"], language="plaintext")

                # Display feature importances for Decision Tree, Random Forest, XGBoost
                if "feature_importances" in info:
                    st.markdown(f"**Feature Importances for {selected_regression_model_pt_1}**")
                    importance_df = pd.DataFrame.from_dict(info["feature_importances"], orient='index',
                                                           columns=['Importance'])
                    importance_df['Importance'] = importance_df['Importance'].apply(lambda x: f"{x:.4f}")
                    st.dataframe(importance_df, use_container_width=True)

                # Display optimized hyperparameters for all models
                if "best_params" in info:
                    st.markdown(f"**Optimized Hyperparameters for {selected_regression_model_pt_1}**")
                    st.json(info["best_params"])
                else:
                    st.warning(f"No optimized hyperparameters available for {selected_regression_model_pt_1}.")
            except Exception as e:
                st.error(f"Failed to extract details for {selected_regression_model_pt_1}: {str(e)}")
        with st.expander("Regression Model Takeaways"):
            st.write("""
            - The models demonstrated a **moderate ability** to predict team winning percentage (`pct`) using six features derived from salary cap structure.

                - **Test R² scores** ranged from **0.21 to 0.32**:
                    - Models explained **21% to 32% of the variance** in team performance.

                - **Test RMSE values** ranged between **0.161 and 0.173**:
                    - Predictions deviated from actual team winning percentages by **16% to 17% on average**.

            - **Best Performing Models**:
                - **ElasticNet**, **LASSO**, and **Random Forest** achieved the strongest performance:
                    - ElasticNet: RMSE = 0.162, R² = 0.313
                    - LASSO: RMSE = 0.161, R² = 0.317
                    - Random Forest: RMSE = 0.162, R² = 0.312

            - **Overfitting and Underfitting Behavior**:
                - **Overfitting** occurs when models perform well on training data but poorly on test data:
                    - **XGBoost** showed signs of overfitting:
                        - Although its cross-validation RMSE was decent (0.166), its **test RMSE worsened to 0.173** and **R² dropped to 0.213**, suggesting limited generalization.
                    - **Decision Tree** had almost identical RMSE on train and test sets but with lower R² (0.305), suggesting possible model simplicity or overfitting to structure in small data.

                - **Underfitting** happens when models cannot capture patterns even in training data:
                    - None of the models showed strong signs of underfitting, but **Linear Regression** had the **highest test RMSE (0.1628)** among the linear models and moderate R² (0.304), indicating some missed complexity.
                    - **KNN Regression** had the **lowest original dataset RMSE (0.094)** but a higher test RMSE (0.164) and a lower R² (0.293), hinting at sensitivity to local structure and poor extrapolation.

                - **Good Generalization**:
                    - **Regularized linear models** (LASSO, Ridge, ElasticNet) and **Random Forest** balanced training and test performance well.
                    - Their **low cross-validation std dev (~0.021)** suggests stability and **good generalization**.

            - **Model Behavior Insights**:
                - **Linear models**:
                    - Offer interpretability and consistency.
                    - Regularization helps in reducing noise and controlling variance.

                - **Non-linear models**:
                    - **Random Forest** generalized well with strong test R².
                    - **KNN** is effective on training data but more sensitive to distributional noise.
                    - **XGBoost**, despite its complexity, may be overfitting and could benefit from more tuning or pruning.

            - **Opportunities for Improvement**:
                - Add features related to **injuries, coaching stability, player experience**, or **offensive/defensive ratings**.
                - Perform **more targeted feature selection**, especially for ratio-based variables which may be collinear.
                - For complex models, apply **regularization, pruning, or advanced tuning** to reduce overfitting risk.
            """)
    with tab7:
        st.markdown("#### Predictive Modeling")
        with st.expander("Prediction Input (All 6 Features)"):
            # Compute dynamic bounds from X_train
            feature_bounds = {
                col: {
                    'min': float(X_train[col].min()),
                    'max': float(X_train[col].max()),
                    'default': float(X_train[col].median())
                } for col in X_train.columns
            }

            tab7col1, tab7col2, tab7col3 = st.columns(3)

            with tab7col1:
                f1 = st.number_input(
                    f"Active Cap Hit Proportion (range: {feature_bounds['cap_hit_prop_active']['min']:.2f} – {feature_bounds['cap_hit_prop_active']['max']:.2f})",
                    min_value=feature_bounds['cap_hit_prop_active']['min'],
                    max_value=feature_bounds['cap_hit_prop_active']['max'],
                    value=feature_bounds['cap_hit_prop_active']['default'],
                    step=0.01
                )
                f4 = st.number_input(
                    f"Ratio Cap Hit Per Player Prop (range: {feature_bounds['ratio_cap_hit_per_player_prop']['min']:.2f} – {feature_bounds['ratio_cap_hit_per_player_prop']['max']:.2f})",
                    min_value=feature_bounds['ratio_cap_hit_per_player_prop']['min'],
                    max_value=feature_bounds['ratio_cap_hit_per_player_prop']['max'],
                    value=feature_bounds['ratio_cap_hit_per_player_prop']['default'],
                    step=0.01
                )

            with tab7col2:
                f2 = st.number_input(
                    f"Active Player Count Proportion (range: {feature_bounds['player_count_prop_active']['min']:.2f} – {feature_bounds['player_count_prop_active']['max']:.2f})",
                    min_value=feature_bounds['player_count_prop_active']['min'],
                    max_value=feature_bounds['player_count_prop_active']['max'],
                    value=feature_bounds['player_count_prop_active']['default'],
                    step=0.01
                )
                f5 = st.number_input(
                    f"Ratio Cap Hit Prop (range: {feature_bounds['ratio_cap_hit_prop']['min']:.2f} – {feature_bounds['ratio_cap_hit_prop']['max']:.2f})",
                    min_value=feature_bounds['ratio_cap_hit_prop']['min'],
                    max_value=feature_bounds['ratio_cap_hit_prop']['max'],
                    value=feature_bounds['ratio_cap_hit_prop']['default'],
                    step=0.01
                )

            with tab7col3:
                f3 = st.number_input(
                    f"Cap Hit Per Player Prop Active (range: {feature_bounds['cap_hit_per_player_prop_active']['min']:.2f} – {feature_bounds['cap_hit_per_player_prop_active']['max']:.2f})",
                    min_value=feature_bounds['cap_hit_per_player_prop_active']['min'],
                    max_value=feature_bounds['cap_hit_per_player_prop_active']['max'],
                    value=feature_bounds['cap_hit_per_player_prop_active']['default'],
                    step=0.01
                )
                f6 = st.number_input(
                    f"Ratio Player Count Prop (range: {feature_bounds['ratio_player_count_prop']['min']:.2f} – {feature_bounds['ratio_player_count_prop']['max']:.2f})",
                    min_value=feature_bounds['ratio_player_count_prop']['min'],
                    max_value=feature_bounds['ratio_player_count_prop']['max'],
                    value=feature_bounds['ratio_player_count_prop']['default'],
                    step=0.01
                )

            input_data = pd.DataFrame({
                'cap_hit_prop_active': [f1],
                'player_count_prop_active': [f2],
                'cap_hit_per_player_prop_active': [f3],
                'ratio_cap_hit_per_player_prop': [f4],
                'ratio_cap_hit_prop': [f5],
                'ratio_player_count_prop': [f6]
            })

            predictions = {}

            if st.button('Predict with All Models'):
                for model_name, model in models.items():
                    try:
                        prediction = model.predict(input_data)
                        # Ensure the prediction is a scalar by flattening arrays
                        if isinstance(prediction, (list, np.ndarray)):
                            prediction = np.array(prediction).flatten()[0]  # Flatten and take the first element
                        # Check if the prediction is a scalar (including NumPy scalars) and can be converted to a float
                        if not (np.isscalar(prediction) and np.isreal(prediction)):
                            raise ValueError(f"Prediction is not a numeric scalar: {prediction}")
                        predictions[model_name] = float(prediction)  # Convert to float for consistency
                    except Exception as e:
                        predictions[model_name] = f"Error: {str(e)}"

                st.markdown("### Prediction Results (Winning Percentage, pct)")
                results_df = pd.DataFrame.from_dict(predictions, orient='index', columns=['Predicted Winning %'])
                results_df['Predicted Winning %'] = results_df['Predicted Winning %'].apply(
                    lambda x: f"{x:.4f}" if isinstance(x, (int, float)) else x
                )
                st.dataframe(results_df, use_container_width=True)

                if all(isinstance(pred, (int, float)) for pred in predictions.values()):
                    max_model = max(predictions, key=predictions.get)
                    min_model = min(predictions, key=predictions.get)
                    st.write(f"**Highest Prediction**: {max_model} ({predictions[max_model]:.4f})")
                    st.write(f"**Lowest Prediction**: {min_model} ({predictions[min_model]:.4f})")
                else:
                    st.warning("Some models failed to predict. Check the errors above.")

        with st.expander("View Bar Chart"):
            if predictions:
                fig_bar = go.Figure(data=[
                    go.Bar(x=list(predictions.keys()), y=list(predictions.values()),
                           text=[f"{v:.4f}" if isinstance(v, (int, float)) else v for v in predictions.values()],
                           textposition='auto')
                ])
                fig_bar.update_layout(
                    title="Model Predictions for Winning Percentage (pct)",
                    xaxis_title="Model",
                    yaxis_title="Predicted Winning %",
                    height=500,
                    margin=dict(l=40, r=40, t=40, b=40)
                )
                st.plotly_chart(fig_bar, use_container_width=True)
            else:
                st.write("No predictions available. Please click 'Predict with All Models' to generate results.")

        with st.expander("View Radar Chart"):
            numeric_predictions = {k: v for k, v in predictions.items() if isinstance(v, (int, float))}
            if numeric_predictions:
                fig_radar = go.Figure(data=go.Scatterpolar(
                    r=list(numeric_predictions.values()),
                    theta=list(numeric_predictions.keys()),
                    fill='toself',
                    text=[f"{v:.4f}" for v in numeric_predictions.values()],
                    hovertemplate="Model: %{theta}<br>Prediction: %{r:.4f}<extra></extra>"
                ))
                fig_radar.update_layout(
                    polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                    title="Radar Chart of Model Predictions",
                    height=500,
                    margin=dict(l=40, r=40, t=40, b=40)
                )
                st.plotly_chart(fig_radar, use_container_width=True)
            else:
                st.write("No numeric predictions available for radar chart. Please click 'Predict with All Models' to generate results.")

        # Visualization: Surface Plots with Checkbox to Toggle cmin/cmax
        with st.expander("View Feature Impact on Winning Percentage (pct)"):
            st.write("These 3D surface plots show how predicted winning percentage (pct) changes as two selected features vary for each model. All axes use the actual data range from the training set.")
            # Select X and Y features
            feature_options = list(X_train.columns)
            x_feature = st.selectbox("Select X-axis Feature", feature_options, index=0)
            y_feature = st.selectbox("Select Y-axis Feature", [f for f in feature_options if f != x_feature], index=0)
            fixed_features = [f for f in feature_options if f not in [x_feature, y_feature]]

            feature_bounds = {
                col: {
                    'min': float(X_train[col].min()),
                    'max': float(X_train[col].max()),
                    'default': float(X_train[col].median())
                } for col in feature_options
            }

            fixed_values = {}
            for feat in fixed_features:
                bounds = feature_bounds[feat]
                fixed_values[feat] = st.slider(
                    f"Set value for {feat}",
                    min_value=bounds['min'],
                    max_value=bounds['max'],
                    value=bounds['default'],
                    step=0.01
                )

            use_cmin_cmax = st.checkbox(
                "Force colorbar range to [0, 1] (shows all ticks but may reduce color variation)", value=False)

            x_range = np.linspace(feature_bounds[x_feature]['min'], feature_bounds[x_feature]['max'], 100)
            y_range = np.linspace(feature_bounds[y_feature]['min'], feature_bounds[y_feature]['max'], 100)
            x_grid, y_grid = np.meshgrid(x_range, y_range)

            grid_data = pd.DataFrame({feat: fixed_values.get(feat, 0.5) for feat in feature_options},
                                     index=range(x_grid.size))
            grid_data[x_feature] = x_grid.ravel()
            grid_data[y_feature] = y_grid.ravel()

            custom_colorscale = [[0.0, 'blue'], [0.5, 'white'], [1.0, 'red']]

            surface_predictions = {}
            for model_name, model in models.items():
                try:
                    # Check if model is a GridSearchCV or pipeline object with preprocessing
                    if hasattr(model, 'predict'):
                        preds = model.predict(grid_data)
                    else:
                        raise ValueError("Model does not have a predict method")
                    preds = np.clip(preds, 0, 1)
                    surface_predictions[model_name] = preds.reshape(x_grid.shape)
                except Exception as e:
                    st.warning(f"Could not generate surface plot for {model_name}: {str(e)}")
                    surface_predictions[model_name] = None

            for model_name, surface_data in surface_predictions.items():
                if surface_data is not None:
                    surface_kwargs = dict(
                        colorscale=custom_colorscale,
                        showscale=True,
                        colorbar=dict(
                            title='Predicted Winning %',
                            tickvals=[i / 10 for i in range(11)],
                            ticktext=[f"{i / 10:.1f}" for i in range(11)],
                            len=0.7,
                            y=0.5,
                            ticks='outside',
                            tickfont=dict(size=12)
                        )
                    )
                    if use_cmin_cmax:
                        surface_kwargs.update(cmin=0.0, cmax=1.0)

                    fig_surface = go.Figure(data=[
                        go.Surface(
                            x=x_grid,
                            y=y_grid,
                            z=surface_data,
                            **surface_kwargs
                        )
                    ])
                    fig_surface.update_layout(
                        title=f"{model_name}: Winning % vs {x_feature} and {y_feature}",
                        scene=dict(
                            xaxis_title=x_feature,
                            yaxis_title=y_feature,
                            zaxis_title='Predicted Winning %',
                            xaxis=dict(range=[x_range.min(), x_range.max()]),
                            yaxis=dict(range=[y_range.min(), y_range.max()]),
                            zaxis=dict(range=[0, 1])
                        ),
                        height=600,
                        margin=dict(l=40, r=40, t=40, b=40)
                    )
                    st.plotly_chart(fig_surface, use_container_width=True)
                else:
                    st.write(f"No surface plot available for {model_name} due to prediction errors.")

    with tab8:
        st.write("""
            ### Part 1 Summary

            - **Winning percentage (`pct`) increases** when teams:
                - Spend a **greater share of their cap on active players** (`cap_hit_prop_active`)
                - Have **smaller, more concentrated active rosters** (`player_count_prop_active`)

            - **Roster efficiency is key to success**:
                - Teams that invest heavily in a **high-value core** tend to win more.
                - **Correlation and clustering analyses** show that elite teams have:
                    - High `cap_hit_prop_active` (80–95%)
                    - Low `player_count_prop_active` (35–45%)
                - Strategy: Focus cap space on **durable, high-performing players** to limit active-inactive transitions.

            - **Statistical correlation findings**:
                - `pct` vs. `cap_hit_prop_active`: **ρ = 0.49**
                - `pct` vs. `log_ratio_cap_hit_per_player_prop`: **ρ = 0.51**
                - `season` vs. `player_count_prop_active`: **ρ = -0.71**
                    - Indicates a **league-wide shift toward leaner active rosters** over time.

            - **Clustering insights (KMeans, GMM, DBSCAN)**:
                - **Top-performing clusters** (e.g., KMeans 0, GMM 4, DBSCAN -1):
                    - Small active rosters
                    - High cap investment efficiency
                    - High win % and net points
                - **Underperforming clusters**:
                    - High inactive burden
                    - Low `cap_hit_prop_active` and poor net points
                - **Balanced/full participation strategies** often led to **average performance**

            - **Regression model results**:
                - Supervised learning models used six features derived from cap/roster structure:
                    - `cap_hit_prop_active`, `cap_hit_per_player_prop_active`, `player_count_prop_active`
                    - `ratio_cap_hit_per_player_prop`, `ratio_cap_hit_prop`, `ratio_player_count_prop`
                - **Test R² values ranged from 0.21 to 0.32**:
                    - Models explained **21% to 32% of variance** in team success.
                - **Test RMSE ranged from 0.161 to 0.173**:
                    - Prediction errors averaged **16–17%** from actual win rates.

            - **Feature importance insights from tree-based models**:
                - All three models emphasized **`ratio_cap_hit_per_player_prop`** as the **most predictive feature**:
                    - **Decision Tree**: 82.8% of total importance
                    - **Random Forest**: 57.4%
                    - **XGBoost**: 57.4%
                - **`cap_hit_prop_active`** and **`ratio_cap_hit_prop`** also ranked highly across models.
                - **`player_count_prop_active`** had very low or **zero importance** in all three tree-based models, suggesting it contributes little once other features are considered.

                **Interpretation**:
                > Tree-based models confirm that **cap efficiency per player** is the most powerful predictor,  
                > while raw roster size metrics (like `player_count_prop_active`) may hold minimal predictive value when used alongside more nuanced features.

            - **Takeaway**: Cap strategy is a **stronger predictor of team success** than roster size alone. Concentrated spending on high-performing active players consistently drives winning outcomes.

            """)

        st.write('---')

        st.write("""
            ### Part 2 Preview

            - Extend the analysis by introducing **positional groupings** (offense, defense, special teams)
                - Map player positions using Spotrac’s taxonomy to categorize roster structure by unit
            - Re-run clustering, correlation, and regression using these unit-level proportions
                - Explore questions like:
                    - Does cap allocation to the **defensive unit** predict success better than offense?
                    - Are **special teams investments** correlated with higher win percentages?
            - Anticipated value:
                - Highlight which **unit-level investments** matter most
                - Discover new high-performing roster strategies that go beyond team-wide aggregates
            """)



if __name__ == "__main__":
    main()
