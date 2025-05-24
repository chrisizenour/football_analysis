# streamlit run streamlit_apps/salary_cap_analysis_pt1.py

import streamlit as st
import joblib
import json
import pandas as pd
import numpy as np
import scipy.stats as stats
from sklearn.tree import export_text
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
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
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
    spotrac_salary_cap_data_df = load_spotrac_dataset(project_data_exports_path)
    nfl_season_records_df = load_nfl_season_records_dataset(project_data_exports_path)
    spotrac_nfl_records_df = load_spotrac_nfl_records_dataset(project_data_exports_path)
    spotrac_nfl_team_season_roster_df = load_spotrac_nfl_team_season_roster_df_dataset(project_data_exports_path)
    spotrac_nfl_team_season_roster_wide_df = load_spotrac_nfl_team_season_roster_wide_df_dataset(project_data_exports_path)
    kmc_labeled_df = load_kmc_labeled_df_dataset(project_data_exports_path)
    kmc_grouped_clusters_labeled_df = load_kmc_grouped_clusters_labeled_df_dataset(project_data_exports_path)
    gmm_labeled_df = load_gmm_labeled_df_dataset(project_data_exports_path)
    gmm_grouped_clusters_labeled_df = load_gmm_grouped_clusters_labeled_df_dataset(project_data_exports_path)

    supervised_learning_pt_1_model_results_df = load_supervised_learning_model_results_pt_1_df_dataset(project_data_exports_path)

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
            title="Sum of cap_hit_pct_league_cap Salary Cap by Team per Season",
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
            title="Sum of cap_hit_pct_league_cap by Team and Roster Status per Season",
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

        st.write("""
            - The average winning % for the NFL has remained essentially unchanged, 0.500, from 2011 to 2024
            - Generally half the teams are above 0.500 and half are below 0.500
            """)

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
            st.write("""
            - Season-to-season, there is stability in team performance as measured by `pct`
            - Individual seasons show little variation and are emblematic of the overall dataset
            - Given consistent nature of `pct`:
                - What effects do changing salary cap allocations have on team performance as measured by `pct`?
                
            """)

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
            st.write("""
            - Notes on dataframe:
                - `roster_status`: 
                    - Engineered field that represents the portion of the roster that the player is on.
                    - Derived from the table that the player appeared in, table 0 = active, table 1+ = inactive
                - `player_count`:
                    - Engineered field that represents the number of players on the roster that were either active or inactive players.
                - `cap_hit_sum`:
                    - Engineered field that represents the sum of the `cap_hit` values for the players on the roster that were either active or inactive players.
                - `player_count_prop`:
                    - Engineered field that represents the proportion of players on the roster that were either active or inactive players.
                - `cap_hit_prop`:
                    - Engineered field that represents the proportion of the `cap_hit_sum` is of that team-season-roster_status combination.
                    - 'active' + 'inactive' = 100% of that team-season-roster_status combination's `cap_hit_sum`.
            """)

        with st.expander("Correlation Amongst Features"):
            # st.dataframe(spotrac_nfl_team_season_roster_wide_df[['season', 'player_count_prop_active', 'cap_hit_prop_active']].corr())

            corr_mat_df_pt_1 = correlation_matrix(spotrac_nfl_team_season_roster_wide_df[['season', 'player_count_prop_active', 'cap_hit_prop_active', 'pct']])
            pval_mat_df_pt_1 = p_val_matrix(spotrac_nfl_team_season_roster_wide_df[['season', 'player_count_prop_active', 'cap_hit_prop_active', 'pct']])

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
            st.write("""
            - Moderate, positive linear correlation between `pct` and `cap_hit_prop_active`
                - Team winning percentage increases as the proportion of the salary cap spent on the active roster increases
            - No linear relationship between `pct` and `season`
                - Corroborates violin plots on Team Record Data tab
            - Moderate, positive linear correlation between `player_count_prop_active` and `cap_hit_prop_active`
                - As the proportion of a team's players on the active roster increases the proportion of a team's salary cap going to the active roster increases
            - Moderate, negative linear correlation between `cap_hit_prop_active` and `season`
                - The proportion of a team's salary cap going to the active roster decreases over time (as the seasons increase from 2011 to 2024) 
            - Moderate-to-almost strong negative linear correlation between `player_count_prop_active` and `season`
                - The proportion of a team's players on the active roster decreases over time (as the seasons increase from 2011 to 2024)
            """)

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
                    colorscale=[[0.0, 'red'], [0.5, 'white'], [1.0, 'blue']],
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
                    colorscale=[[0.0, 'red'], [0.5, 'white'], [1.0, 'blue']],
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
            - Overall, the change in salary cap proportion allocations have not resulted in a change league winning percentages at the overall dataset level or on a season-to-season basis
            """)

    with tab5:
        st.markdown("#### Team - Season - Roster Status Unsupervised Learning")
        st.write("""
        What underlying structures are there to the Spotrac salary cap data?
        """)
        with st.expander("Methodology"):
            st.write("""
            - Dataset for analysis was the spotrac_nfl_team_season_roster_df
            - Dataframe is in long format, so each team-season combination had two rows, one for each `roster_status` (active and inactive)
                - `player_count_prop` and `cap_hit_prop` fields for active and inactive roster statuses add up to 1.0, so only the rows with the active roster status are used
            - In addition to `player_count_prop` and `cap_hit_prop`, the `season` field is also used
            - The resulting dataset used for unsupervised learning contain 448 observations and 3 columns (`season`, `player_count_prop`, and `cap_hit_prop`)
            - Three different unsupervised learning models were used:
                - KMeans clustering
                - Gaussian Mixture Model
                - DBSCAN clustering
            - After performing unsupervised learning, cluster assignments for each row are applied to the original dataset, spotrac_nfl_team_season_roster_df
            - The original dataset with cluster assignments is then grouped by cluster and the mean values for each cluster are calculated 
            """)

        with st.expander("Original and Filtered Dataset used for clustering"):
            st.write("Original Dataset: spotrac_nfl_team_season_roster_df")
            st.dataframe(spotrac_nfl_team_season_roster_df)
            st.write("---")
            st.write("Clustering Dataset: Filtered spotrac_nfl_team_season_roster_df")
            st.dataframe(spotrac_nfl_team_season_roster_df.loc[spotrac_nfl_team_season_roster_df['roster_status'] == 'active', ['season', 'player_count_prop', 'cap_hit_prop', ]])

        with st.expander("KMeans Clustering"):
            st.write("""
            - Elbow Plot and Average Cluster Silhouette Score plot indicate 4 clusters as optimal cluster quantity
            """)
            st.write('---')
            st.write('spotrac_nfl_team_season_roster_df with KMeans Cluster Assignments')
            kmc_labeled_df_clusters = kmc_labeled_df['label'].unique()
            kmc_labeled_df_clusters_choice = st.multiselect('Select KMeans Cluster(s) to View', kmc_labeled_df_clusters, default=kmc_labeled_df_clusters)
            st.dataframe(kmc_labeled_df.loc[kmc_labeled_df['label'].isin(kmc_labeled_df_clusters_choice), :])
            st.write("---")

            kmc_labeled_df_numerical_cols = ['season', 'player_count', 'cap_hit_sum', 'player_count_prop', 'cap_hit_prop',
    'w', 'l', 'pct', 'pf', 'pa', 'net_pts', 'div_win_pct', 'conf_win_pct',
    'pc_1', 'pc_2',]
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


            st.write('Cluster means for spotrac_nfl_team_season_roster_df')
            st.dataframe(kmc_grouped_clusters_labeled_df)
            st.write("---")
            st.write("""
            Observations concerning the cluster means dataframe:
            - Cluster 0:
                - Superior performance relative to other clusters as measured by pct and other metrics
                - On average, 81% of annual salary cap expenditures on the active roster
                - On average, 39% of players that register a cap hit are on the active roster
                - On average, 60% overall winning percentage
            - Cluster 2:
                - Better `cap_hit_prop` and `player_count_prop` values for the active roster than of Cluster 0
                    - On average, 87% of salary cap expenditures go toward the active roster (81% for Cluster 0)
                    - On average, 69% of players that register a cap hit are on the active roster (39% for Cluster 0)
                - Overall winning percentage is roughly equal to Average League winning percentage over the entire dataset
            - Potential explanations for difference between Cluster 0 and Cluster 2:
                - On average, Cluster 0 scored ~45 more points per season than did Cluster 2; hinting at better scoring abilities
                - On average, Cluster 2 did a better job keeping salary cap affecting players on the active roster (69% vs 39%)
            """)
        
        with st.expander("Gaussian Mixture Model (GMM) Clustering"):
            st.write("""
            - BIC and AIC vs Number of Components curves indicate 4 clusters as optimal cluster quantity
            """)
            st.write('spotrac_nfl_team_season_roster_df with GMM Cluster Assignments')
            gmm_labeled_df_clusters = gmm_labeled_df['label'].unique()
            gmm_labeled_df_clusters_choice = st.multiselect('Select GMM Cluster(s) to View', gmm_labeled_df_clusters,
                                                            default=gmm_labeled_df_clusters)
            st.dataframe(gmm_labeled_df.loc[gmm_labeled_df['label'].isin(gmm_labeled_df_clusters_choice), :])
            st.write("---")

            gmm_labeled_df_numerical_cols = ['season', 'player_count', 'cap_hit_sum', 'player_count_prop',
                                             'cap_hit_prop',
                                             'w', 'l', 'pct', 'pf', 'pa', 'net_pts', 'div_win_pct', 'conf_win_pct',
                                             'pc_1', 'pc_2', ]
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
            st.write("""
            Observations concerning the cluster means dataframe:
            - Cluster 0:
                - Superior performance relative to other clusters as measured by pct and other metrics
                    - On average, 82% of annual salary cap expenditures on the active roster
                    - On average, 39% of players that register a cap hit are on the active roster
                    - On average, 62% overall winning percentage
            - Cluster 2:
                - Second best performance relative to other clusters as measured by pct and other metrics
                    - On average, 85% of annual salary cap expenditures on the active roster
                    - On average, 70% of players that register a cap hit are on the active roster
                - Overall winning percentage is roughly equal to Average League winning percentage over the entire dataset
            - Potential explanations for difference between Cluster 0 and Cluster 2:
                - On average, Cluster 0 scored ~60 more points per season than did Cluster 2; hinting at better scoring abilities
                - On average, Cluster 0 allowed ~8 fewer point per season than did Cluster 2; hinting at better defensive abilities
                - On average, Cluster 2 did a better job keeping salary cap affecting players on the active roster (70% vs 39%)
            """)

        with st.expander("DBSCAN Clustering"):
            st.write("""
            DBSCAN clustering was performed, but the algorithm struggled with the dataset and returned all datapoints as belonging to the same, single cluster
            """)

        with st. expander("Clustering Takeaways"):
            st.write("""
            - `cap_hit_prop` and `player_count_prop` hint at better team performance outcomes, but are confounded
                - On average, teams with 80%+ of annual salary cap expenditures on the active roster had both superior and average performance outcomes as measured by `pct` and other scoring metrics
                    - One difference between these samples is that the Cluster 2s had a higher proportion of players on the active roster (~69%) than did Cluster 0s (~39%)
                    - Cluster 2s had, on average, more players (~57) on the active roster than did Cluster 0s (~53), 
                    and Cluster 2s proportionally (~31% vs ~61%) had fewer players on the inactive roster. So those 
                    teams' active players may have greater durability, utilize better training and recovery techniques, and are potentially overpaid given scoring performance metrics
            - Cluster 0 for both approaches considered to be the cluster of superior performance
                - KMC average `pct`: 60%, GMM average `pct`: 62%
                - KMC average `cap_hit_prop`: 81%, GMM average `cap_hit_prop`: 82%
                - KMC average `player_count_prop`: 39%, GMM average `player_count_prop`: 39%
                - KMC average `net_points`: +50, GMM average `net_points`: +64
                - KMC average `pf`: 407 (~45 more than second best cluster), GMM average `pf`: 415 (~60 more than second best cluster)
                - KMC average `pa`: 356 (~3 less than second best cluster), GMM average `pa`: 352 (~8 less than second best cluster)
            - GMM included fewer observations in Cluster 0 than did KMC (83 vs 125), though they both had the same count for Cluster 2, 60
                - All 83 observations from GMM Cluster 0 appear in KMC's Cluster 0
                - KMC and GMM shared 54 of 60 (90%) observations that each labeled for inclusion into Cluster 2
            """)

    with tab6:
        st.markdown("#### Team - Season - Roster Status Supervised Learning")
        st.write("""
        Predicting a team's season winning percentage based on the proportion of its salary cap allocated to players 
        on its active roster and the proportion of its players on the active roster.
        """)
        with st.expander("Methodology"):
            st.write("""
            - Dataset for analysis was the spotrac_nfl_team_season_roster_df
            - Dataframe is in long format, so each team-season combination had two rows, one for each `roster_status` (active and inactive)
                - `player_count_prop` and `cap_hit_prop` fields for active and inactive roster statuses add up to 1.0, so only the rows with the active roster status are used
            - The resulting dataset used for supervised learning contain 448 observations and 3 columns (`pct`, 
            `player_ount_prop`,  and `cap_hit_prop`)
                - `season` was not used as it would prevent out-of-sample predictions from being performed
            - The dataset was split into two subsets: 
                - X: the independent variables `cap_hit_prop` and `player_count_prop`
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
                - For example, the Decision Tree Regression model provides a decision tree, Decision Tree and Random 
                Forest provide feature importance information, and Ridge, LASSO, ElasticNet Regression models adjust 
                coefficients.
            - When training the models, scikit-learn's GridSearchCV function was used to find the optimal hyperparameters
            - After generating predictions using the test set, plots were generated to ascertain the ability of the 
            models to predict the dependent variable, `pct`
            - All models were then tested on the full 448 observation dataset to provide a final assessment of model 
            performance.
            """)
        with st.expander('Original and Filtered Dataset used for regression model training'):
            st.write("Original Dataset: spotrac_nfl_team_season_roster_df")
            st.dataframe(spotrac_nfl_team_season_roster_df)
            st.write('---')
            st.write('Regression model training dataset: Filtered spotrac_nfl_team_season_roster_df')
            st.dataframe(spotrac_nfl_team_season_roster_df.loc[
                             spotrac_nfl_team_season_roster_df['roster_status'] == 'active', ['pct',
                                                                                              'player_count_prop',
                                                                                              'cap_hit_prop', ]])
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


            feature_names_pt_1 = ['player_count_prop', 'cap_hit_prop']
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

            original_feature_names_pt_1 = ['cap_hit_prop', 'player_count_prop']

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
            - Models generally struggled predicting team winning percentage based on `cap_hit_prop` and `player_count_prop` though they were within ~2% of each other
                - Models' Test Dataset RMSE values ranged from 0.1614 to 0.1888
                    - i.e., Predicted winning percentage were between 16% and 19% off from actual values
                - XGBoost, Decision Tree, and Random Forest exhibit slight overfitting
                - Linear Regression, Ridge, LASSO, and ElasticNet exhibit slight underfitting
            - Model performance may be improved by adjusting the hyperparameters for each model
            - Adding additional features may assist models in identifying patterns within the data that aid in predicting winning percentage
                
                """)
    with tab7:
        st.markdown("#### Predictive Modeling")
        with st.expander("Model Predictions"):
            tab7col1, tab7col2 = st.columns(2)
            with tab7col1:
                active_cap_hit_prop_choice = st.number_input('Enter Active Cap Hit Proportion (0 - 1)', min_value=0.0,
                                                   max_value=1.0, value=0.8, step=0.01)
            with tab7col2:
                active_player_count_prop_choice = st.number_input('Enter Active Player Count Proportion (0 - 1)',
                                                              min_value=0.0, max_value=1.0, value=0.8, step=0.01)
            input_data = pd.DataFrame({
                'cap_hit_prop': [active_cap_hit_prop_choice],
                'player_count_prop': [active_player_count_prop_choice]
            })

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
            st.write("These 3D surface plots show how predicted winning percentage (pct) changes as `cap_hit_prop` and `player_count_prop` vary for each model. All axes range from 0 to 1, with blue at 0.0, white at 0.5, and red at 1.0.")

            # Checkbox to toggle cmin and cmax
            use_cmin_cmax = st.checkbox("Force colorbar range to [0, 1] (shows all ticks but may reduce color variation)", value=False)

            # Create a grid of values for cap_hit_prop and player_count_prop, ensuring range 0 to 1
            cap_hit_range = np.linspace(0, 1, 100)
            player_count_range = np.linspace(0, 1, 100)
            cap_hit_grid, player_count_grid = np.meshgrid(cap_hit_range, player_count_range)

            # Flatten the grids for prediction
            grid_data = pd.DataFrame({
                'cap_hit_prop': cap_hit_grid.ravel(),
                'player_count_prop': player_count_grid.ravel()
            })

            # Define custom colorscale: blue (0.0) to white (0.5) to red (1.0)
            custom_colorscale = [[0.0, 'blue'], [0.5, 'white'], [1.0, 'red']]

            # Predict for each model across the grid
            surface_predictions = {}
            for model_name, model in models.items():
                try:
                    preds = model.predict(grid_data)
                    # Clip predictions to [0, 1] to match the requested range
                    preds = np.clip(preds, 0, 1)
                    # Reshape predictions back to grid shape
                    surface_predictions[model_name] = preds.reshape(cap_hit_grid.shape)
                except Exception as e:
                    st.warning(f"Could not generate surface plot for {model_name}: {str(e)}")
                    surface_predictions[model_name] = None

            # Create a surface plot for each model with additional ticks and toggled cmin/cmax
            for model_name, surface_data in surface_predictions.items():
                if surface_data is not None:
                    # Configure surface plot based on checkbox
                    if use_cmin_cmax:
                        # Force colorbar range to [0, 1]
                        surface_kwargs = dict(
                            cmin=0.0,
                            cmax=1.0,
                            colorscale=custom_colorscale,
                            showscale=True,
                            colorbar=dict(
                                title='Predicted Winning %',
                                tickvals=[i / 10 for i in range(11)],  # Ticks at 0.0, 0.1, 0.2, ..., 1.0
                                ticktext=[f"{i/10:.1f}" for i in range(11)],  # Labels as 0.0, 0.1, ..., 1.0
                                len=0.7,
                                y=0.5,
                                ticks='outside',
                                tickfont=dict(size=12)
                            )
                        )
                    else:
                        # Let Plotly auto-scale the colorscale
                        surface_kwargs = dict(
                            colorscale=custom_colorscale,
                            showscale=True,
                            colorbar=dict(
                                title='Predicted Winning %',
                                tickvals=[i / 10 for i in range(11)],  # Ticks at 0.0, 0.1, 0.2, ..., 1.0
                                ticktext=[f"{i/10:.1f}" for i in range(11)],  # Labels as 0.0, 0.1, ..., 1.0
                                len=0.7,
                                y=0.5,
                                ticks='outside',
                                tickfont=dict(size=12)
                            )
                        )

                    fig_surface = go.Figure(data=[
                        go.Surface(
                            x=cap_hit_grid,
                            y=player_count_grid,
                            z=surface_data,
                            **surface_kwargs
                        )
                    ])
                    fig_surface.update_layout(
                        title=f"{model_name}: Winning Percentage (pct) vs Cap Hit Prop and Player Count Prop",
                        scene=dict(
                            xaxis_title='Cap Hit Proportion',
                            yaxis_title='Player Count Proportion',
                            zaxis_title='Predicted Winning %',
                            xaxis=dict(range=[0, 1]),
                            yaxis=dict(range=[0, 1]),
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
        Part 1:
        - Teams' winning percentage (`pct`) increased as the proportion of the salary cap spent on the active roster increased and proportion of players on the inactive roster decreased
            - Minimizing the incidence of players moving from active to inactive roster is key
                - Teams should optimize training and recover strategies
                - Players selection should include analysis of player's ability to remain available
                - Where the proportion of players on the active roster is high and `pct` is low, the team may have players that are not as productive as players on successful teams, players not fit for the coaching system, or something else
        - Season over season, the NFL has a consistent average winning percentage of approximately .500
        - Over the time studied timespan, 2011 - 2024, as the average winning percentage of the NFL remained at 0.500 the proportion of salary cap spent on the active roster decreased while the proportion spent on the inactive roster increased
            - *Keep in mind for future area of analysis outside of this current project pipeline*
        - Moderate positive linear relationship between winning percentage and `cap_hit_prop_active` (0.49)
        - Moderate negative linear relationship between season and `player_count_prop_active` (-0.71)
        - KMeans and GMM clustering algorithms found similar clusters (Cluster 0) that exhibited superior performance
        - Regression models generally struggled predicting team winning percentage
            - Tree-based models exhibited slight overfitting
            - Non-tree-based models exhibited slight underfitting
            - Adding additional features may help models better predict winning percentage
        - Tree-based models consistently found the proportion of the salary cap spent on the active roster significantly contributes to model predictions of winning percentage
            - Decision Tree: `cap_hit_prop`: 0.8276, `player_count_prop`: 0.1724
            - Random Forest: `cap_hit_prop`: 0.8301, `player_count_prop`: 0.1699
            - XGBoost: `cap_hit_prop`: 0.6247, `player_count_prop`: 0.3753
        - Compared to Linear Regression's coefficients, Ridge, LASSO, and ElasticNet all shrank the coefficients, especially `player_count_prop`. This suggests that while contributory towards predicting `pct`, the proportion of players on the inactive roster has little importance compared to `cap_hit_prop`
        """)
        st.write('---')
        st.write("""
        Part 2
        - Incorporate team offense, defense and special team positional groupings
            - Positions provided by spotrac.com are mapped to offense, defense, or special teams
        - Perform same analyses as Part 1, but with positional groupings as a categorical label
        """)



if __name__ == "__main__":
    main()
