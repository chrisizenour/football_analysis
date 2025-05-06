# streamlit run streamlit_apps/salary_cap_analysis_pt1.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly_express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import sys
# Ensure the top-level project directory is in the Python path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from path_config import project_path, project_data_sources_path, project_data_exports_path

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

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        'Spotrac Data',
        'Team Record Data',
        'Spotrac Data + Team Record Data',
        'Team - Season - Roster Status EDA',
        'Team - Season - Roster Status Unsupervised Learning'
    ])

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
            - 'cap_hit': This represents the dollar amount spent by teams on players that counts toward their salary cap.
            - 'pos': This represents the position of the player.
            - 'team': This represents the team that the player plays for.
            - 'season': This represents the year of the season.
            - 'table_number': This represents the number of the table from Spotrac the player is on.
                - table 0 is the active roster
                - table 1 and greater is the inactive roster
            - 'position_level_one': This is an engineered column the represents the highest positional grouping of the player (offense, defense, special team)
            - 'position_level_two': This is an engineered column the represents the second highest positional grouping of the player (offensive_line, defensive_line, running_back, etc.)
            
            - 'cap_hit_pct_league_cap': This represents the percentage of the salary cap that the player consumed of that season's team salary cap.
                - When summed, teams can have greater than 100% of the season's salary cap.
                - Multiple instances of teams with greater than 100% of the salary cap are possible.
                - Follow-on analysis will use the 'cap_hit' value and consider the each team's salary cap allocations to be 100% and the constituent proportions of the salary cap as the cap hit percentages.
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
            - Season-to-season, there is stability in team performance as measured by pct
            - Individual seasons show little variation and are emblematic of the overall dataset
            - Given consistent nature of pct:
                - What effects do changing salary cap allocations have on team performance as measured by pct?
                
            """)

        with st.expander("Future uses of this dataset"):
            st.write("""
            For this analysis, I will use the following columns:
            - 'nfl_team_name': This represents the name of the team.
            - Team outcome (performance) metrics:
                - 'pct': This represents the winning % of the team.
                - 'w': This represents the number of wins of the team.
                - 'l': This represents the number of losses of the team.
                - 'pf': This represents the number of points for of the team.
                - 'pa': This represents the number of points against of the team.
                - 'net_pts': This represents the net points scored by the team.
                - 'div_win_pct': This represents the winning % of the team in the division.
                - 'conf_win_pct': This represents the winning % of the team in the conference.
            - This analysis's initial line of effort will focus on the team's overall season winning percentage (pct).
            
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
            st.write("""
            - Notes on dataframe:
                - 'roster_status': 
                    - Engineered field that represents the portion of the roster that the player is on.
                    - Derived from the table that the player appeared in, table 0 = active, table 1+ = inactive
                - 'player_count':
                    - Engineered field that represents the number of players on the roster that were either active or inactive players.
                - 'cap_hit_sum':
                    - Engineered field that represents the sum of the cap_hit values for the players on the roster that were either active or inactive players.
                - 'player_count_prop':
                    - Engineered field that represents the proportion of players on the roster that were either active or inactive players.
                - 'cap_hit_prop':
                    - Engineered field that represents the proportion of the cap_hit_sum is of that team-season-roster_status combination.
                    - 'active' + 'inactive' = 100% of that team-season-roster_status combination's cap_hit_sum.
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
            - Dataframe is in long format, so each team-season combination had two rows, one for each roster status (active and inactive)
                - player_count_prop and cap_hit_prop fields for active and inactive roster statuses add up to 1.0, so only the rows with the active roster status are used
            - In addition to player_count_prop and cap_hit_prop, the season field is also used
            - The resulting dataset used for unsupervised learning contain 448 observations and 3 columns (season, player_count_prop, and cap_hit_prop)
            - Three different unsupervised learning models were used:
                - KMeans clustering
                - Gaussian Mixture Model
                - DBSCAN clustering
            - After performing unsupervised learning, cluster assignments for each row are applied to the original dataset, spotrac_nfl_team_season_roster_df
            - The original dataset with cluster assignments is then grouped by cluster and the mean values for each cluster are calculated 
            """)
        with st.expander("Original and Filtered Datasets"):
            st.write("Original Dataset: spotrac_nfl_team_season_roster_df")
            st.dataframe(spotrac_nfl_team_season_roster_df)
            st.write("---")
            st.write("Clustering Dataset: Filtered spotrac_nfl_team_season_roster_df")
            st.dataframe(spotrac_nfl_team_season_roster_df.loc[spotrac_nfl_team_season_roster_df['roster_status'] == 'active', ['season', 'player_count_prop', 'cap_hit_prop', ]])

if __name__ == "__main__":
    main()
