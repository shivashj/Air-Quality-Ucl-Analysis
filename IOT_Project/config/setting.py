"""
config/settings.py

Central configuration file for the IoT Final Project.
All global parameters, paths, and constants are defined here.
"""

from pathlib import Path
import os

# Project Base Directory

BASE_DIRECTORY = Path(__file__).resolve().parent.parent

# Data Paths

DATA_DIR = BASE_DIRECTORY / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
RESULTS_DIR = BASE_DIRECTORY / "results"

# Dataset Configuration
                       
DEFAULT_DATASET_NAME = "AirQualityUCI.csv"
RAW_DATA_FILE = RAW_DATA_DIR / DEFAULT_DATASET_NAME

# Model Parameters

TRAIN_TEST_SPLIT_RATIO = 0.8
RANDOM_STATE = 42

# Logging Configuration

LOG_LEVEL = "INFO"
LOG_FILE_NAME = "project.log"
LOG_FILE_PATH = BASE_DIRECTORY / LOG_FILE_NAME

# Validation Rules

REQUIRED_COLUMNS = [
    "Date",
    "Time",
    "CO(GT)",
    "NO2(GT)",
    "C6H6(GT)",
    "T",
    "RH",
    "AH"
]
   
# Dataset Selection

USE_SYNTHETIC_DATA = True   # True → synthetic | False → real

REAL_DATA_FILE = RAW_DATA_DIR / "AirQualityUCI.csv"
SYNTHETIC_DATA_FILE = RAW_DATA_DIR / "AirQualityUCI_synthetic.csv"


# Environment Mode

ENVIRONMENT = os.getenv("PROJECT_ENV", "development")
