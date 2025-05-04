# football/path_config.py

from pathlib import Path

# Automatically find the project root (assuming this file stays in the root)
project_path = Path(__file__).resolve().parent

# Define key data directories
project_data_sources_path = project_path / 'data' / 'sources'
project_data_exports_path = project_path / 'data' / 'exports'