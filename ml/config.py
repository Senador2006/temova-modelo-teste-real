from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"
SYNTHETIC_DIR = DATA_DIR / "synthetic"
TEMPLATES_DIR = DATA_DIR / "templates"
SIMULATED_DIR = DATA_DIR / "simulated_field"
ARTIFACTS_DIR = BASE_DIR / "artifacts"
RESULTS_DIR = BASE_DIR / "results"

TRAIN_PATH = SYNTHETIC_DIR / "synthetic_lstm_train.csv"
VAL_PATH = SYNTHETIC_DIR / "synthetic_lstm_validation.csv"
TEST_PATH = SYNTHETIC_DIR / "synthetic_lstm_test.csv"
FULL_SYNTHETIC_PATH = SYNTHETIC_DIR / "synthetic_lstm_full.csv"

DATA_DICTIONARY_PATH = DATA_DIR / "data_dictionary.csv"
REAL_FIELD_TEMPLATE_PATH = TEMPLATES_DIR / "real_field_collection_template.csv"
REAL_INPUT_TEMPLATE_PATH = TEMPLATES_DIR / "real_lstm_input_template.csv"
SIMULATED_FIELD_COLLECTION_PATH = SIMULATED_DIR / "simulated_field_collection.csv"
SIMULATED_REAL_INPUT_PATH = SIMULATED_DIR / "simulated_real_lstm_input.csv"

SEED = 20260813
FIELD_SEED = 20260814
MODEL_VERSION = "2.3.0"

LOOK_BACK = 2
CONTEXT_DAYS = 2
HORIZON_DAYS = 2
N_SEQUENCE_FEATURES = 4
N_FUTURE_FEATURES = 3
LSTM_UNITS = 32
DENSE_UNITS = 16

SEQUENCE_FEATURE_ORDER = [
    "altura_cm",
    "temp_media_c",
    "precipitacao_mm",
    "umidade_media_pct",
]

SEQUENCE_T1 = [
    "altura_t1_cm",
    "temp_t1_c",
    "precipitacao_t1_mm",
    "umidade_t1_pct",
]
SEQUENCE_T2 = [
    "altura_t2_cm",
    "temp_t2_c",
    "precipitacao_t2_mm",
    "umidade_t2_pct",
]
FUTURE_FEATURES = [
    "temp_futuro_2d_c",
    "precipitacao_futuro_2d_mm",
    "umidade_futuro_2d_pct",
]
TARGET = "crescimento_futuro_2d_cm"
PAST_GROWTH = "crescimento_passado_2d_cm"
BASELINE_FORMULA = "altura_t2_cm - altura_t1_cm"

REQUIRED_INPUT_COLUMNS = SEQUENCE_T1 + SEQUENCE_T2 + FUTURE_FEATURES
