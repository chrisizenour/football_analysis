# football/path_config.py

from pathlib import Path

# Automatically find the project root (assuming this file stays in the root)
project_path = Path(__file__).resolve().parent

# Define key data directories
project_data_sources_path = project_path / 'data' / 'sources'
project_data_exports_path = project_path / 'data' / 'exports'

# Define key saved model directories
project_pt_1_models_path = project_path / 'models' / 'pt_1'
project_pt_2_models_path = project_path / 'models' / 'pt_2'
project_pt_3_models_path = project_path / 'models' / 'pt_3'

# Define papers directory
project_papers_path = project_path / 'papers'