from pathlib import Path

WORK_DIR = Path(__file__).parents[1]
CONFIG_DIR = WORK_DIR / 'configs'
DATA_DIR = WORK_DIR / 'data'
VIDEO_DIR = WORK_DIR / 'data' / '.videos'
SOURCE_DIR = WORK_DIR / 'src'
# MODELS_DIR = WORK_DIR / 'src' / 'models'
MODELS_DIR = WORK_DIR
EXPERIMENTS_DIR = WORK_DIR / 'experiments'
