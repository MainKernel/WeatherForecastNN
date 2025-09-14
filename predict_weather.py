import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import os
import logging
from datetime import datetime
import matplotlib.pyplot as plt

# --- Налаштування логування ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
tf.get_logger().setLevel(logging.INFO)

# --- Налаштування GPU (якщо доступно) ---
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logging.info(f"Using GPUs: {gpus}")
    except RuntimeError as e:
        logging.error(f"Error setting up GPUs: {e}")
else:
    logging.info("No GPU devices found, using CPU.")

# --- Constants (мають бути ідентичними до тих, що використовувались при навчанні) ---
DATA_DIRECTORY = 'data/'
MODEL_PATH = 'weather_forecast_model_all_params_72h_encdec.h5'  # Шлях до нової моделі Encoder-Decoder
SEQUENCE_LENGTH = 24
FORECAST_HORIZON = 72

FEATURE_COLUMNS = [
    "ALLSKY_SFC_SW_DWN", "WS2M", "PS", "RH2M", "T2M",
    "SNODP", "TS", "WD10M", "T2MDEW", "PRECTOTCORR"
]
LABEL_COLUMNS = FEATURE_COLUMNS.copy()


# --- Data Processing Class (ідентичний до того, що використовувався при навчанні) ---
class WeatherDataProcessor:
    def __init__(self, data_directory, sequence_length, forecast_horizon, feature_cols, label_cols):
        self.data_directory = data_directory
        self.sequence_length = sequence_length
        self.forecast_horizon = forecast_horizon
        self.feature_cols = feature_cols
        self.label_cols = label_cols

        logging.info(
            "Loading full historical data to fit scalers (required for consistent prediction preprocessing)...")
        self.all_data = self._load_and_parse_csv_data()

        self.feature_scaler = MinMaxScaler()
        self.label_scaler = MinMaxScaler()

        logging.info("Fitting scalers on full historical data...")
        self._fit_scalers()

    def _load_and_parse_csv_data(self):
        all_dfs = []
        for filename in sorted(os.listdir(self.data_directory)):
            if filename.endswith(".csv"):
                filepath = os.path.join(self.data_directory, filename)
                df = pd.read_csv(filepath)
                df['DATETIME'] = pd.to_datetime(df[['YEAR', 'MO', 'DY', 'HR']].astype(str).agg('-'.join, axis=1),
                                                format='%Y-%m-%d-%H')
                all_dfs.append(df)

        full_df = pd.concat(all_dfs, ignore_index=True)
        full_df = full_df.sort_values(by='DATETIME').reset_index(drop=True)

        for col in list(set(self.feature_cols + self.label_cols)):
            if col not in full_df.columns:
                full_df[col] = 0.0
            full_df[col] = full_df[col].replace(-999.0, np.nan)
            full_df[col] = full_df[col].ffill().bfill()
            if full_df[col].isnull().any():
                full_df[col] = full_df[col].fillna(0.0)

        return full_df

    def _fit_scalers(self):
        self.feature_scaler.fit(self.all_data[self.feature_cols])
        self.label_scaler.fit(self.all_data[self.label_cols])

    def normalize_features(self, data_df):
        return self.feature_scaler.transform(data_df[self.feature_cols])

    def denormalize_labels(self, scaled_labels):
        return self.label_scaler.inverse_transform(scaled_labels)


# --- Main Prediction Logic ---
if __name__ == "__main__":
    PREDICTION_DATA_FILE = 'data/TestData.csv'

    # 1. Завантаження моделі
    logging.info(f"Loading trained model from {MODEL_PATH}...")
    try:
        # Моделі Keras Functional API (Model) завантажуються так само, як Sequential
        loaded_model = load_model(MODEL_PATH)
        logging.info("Model loaded successfully.")
    except Exception as e:
        logging.error(f"Failed to load model from {MODEL_PATH}. Error: {e}")
        logging.error(
            "Please ensure the model was trained and saved correctly (e.g., by running train_weather_model.py first).")
        exit()

    # 2. Ініціалізація WeatherDataProcessor для отримання скалерів
    processor = WeatherDataProcessor(DATA_DIRECTORY, SEQUENCE_LENGTH, FORECAST_HORIZON, FEATURE_COLUMNS, LABEL_COLUMNS)

    # 3. Завантаження та попередня обробка нових даних для прогнозу
    logging.info(f"Loading new data for prediction from {PREDICTION_DATA_FILE}...")
    try:
        new_data_df = pd.read_csv(PREDICTION_DATA_FILE)
        new_data_df['DATETIME'] = pd.to_datetime(
            new_data_df[['YEAR', 'MO', 'DY', 'HR']].astype(str).agg('-'.join, axis=1), format='%Y-%m-%d-%H')
        new_data_df = new_data_df.sort_values(by='DATETIME').reset_index(drop=True)

        for col in list(set(FEATURE_COLUMNS + LABEL_COLUMNS)):
            if col not in new_data_df.columns:
                new_data_df[col] = 0.0
            new_data_df[col] = new_data_df[col].replace(-999.0, np.nan)
            new_data_df[col] = new_data_df[col].ffill().bfill()
            if new_data_df[col].isnull().any():
                new_data_df[col] = new_data_df[col].fillna(0.0)

        if len(new_data_df) < SEQUENCE_LENGTH:
            logging.error(
                f"The prediction data file '{PREDICTION_DATA_FILE}' must contain at least {SEQUENCE_LENGTH} hours of data. Found: {len(new_data_df)}")
            exit()

        input_sequence_df = new_data_df.tail(SEQUENCE_LENGTH)

        input_for_prediction_scaled = processor.normalize_features(input_sequence_df)
        input_for_prediction_reshaped = input_for_prediction_scaled.reshape(1, SEQUENCE_LENGTH, len(FEATURE_COLUMNS))
        logging.info(f"Input shape for prediction: {input_for_prediction_reshaped.shape}")

    except Exception as e:
        logging.error(f"Error loading or preprocessing prediction data from {PREDICTION_DATA_FILE}. Error: {e}")
        exit()

    # 4. Виконання прогнозу
    logging.info(f"Making {FORECAST_HORIZON}-hour prediction for all parameters...")
    # Модель тепер поверне 3D тензор: (1, FORECAST_HORIZON, label_size)
    output_prediction_scaled = loaded_model.predict(input_for_prediction_reshaped)

    # 5. Де-нормалізація результатів
    predicted_labels = processor.denormalize_labels(
        output_prediction_scaled.reshape(FORECAST_HORIZON, len(LABEL_COLUMNS)))

    logging.info(f"\n--- {FORECAST_HORIZON}-Hour Weather Forecast (All Parameters) ---")

    last_input_time = new_data_df['DATETIME'].iloc[-1]

    forecast_df = pd.DataFrame(predicted_labels, columns=LABEL_COLUMNS)
    forecast_times = [last_input_time + pd.Timedelta(hours=i + 1) for i in range(FORECAST_HORIZON)]
    forecast_df.insert(0, 'DATETIME', forecast_times)

    logging.info(forecast_df.to_string(index=False))

    # 6. Побудова графіків для всіх прогнозованих параметрів
    num_plots_per_row = 2
    num_rows = int(np.ceil(len(LABEL_COLUMNS) / num_plots_per_row))

    plt.figure(figsize=(18, 6 * num_rows))
    plt.suptitle(f'{FORECAST_HORIZON}-Hour Weather Forecast for All Parameters', fontsize=16)

    for i, param_col in enumerate(LABEL_COLUMNS):
        plt.subplot(num_rows, num_plots_per_row, i + 1)
        plt.plot(forecast_df['DATETIME'], forecast_df[param_col], marker='.', linestyle='-', markersize=4,
                 label=f'Predicted {param_col}')
        plt.title(f'{param_col} Forecast')
        plt.xlabel('Time')
        plt.ylabel(param_col)
        plt.grid(True)
        plt.legend()
        plt.xticks(rotation=45, ha='right')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(f'{FORECAST_HORIZON}h_weather_forecast_all_params_plot.png')
    logging.info(
        f"{FORECAST_HORIZON}-hour weather forecast plot for all parameters saved to {FORECAST_HORIZON}h_weather_forecast_all_params_plot.png")