import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model  # Додано Model API
from tensorflow.keras.layers import LSTM, Dense, Input, TimeDistributed, RepeatVector  # Додано RepeatVector
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
from sklearn.preprocessing import MinMaxScaler
import os
import logging
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

# --- Constants ---
DATA_DIRECTORY = 'data/'
MODEL_PATH = 'weather_forecast_model_all_params_72h_encdec.h5'  # Новий шлях для моделі Encoder-Decoder
SEQUENCE_LENGTH = 24
FORECAST_HORIZON = 72
BATCH_SIZE = 32
LSTM_LAYER_SIZE = 128  # Можливо, краще використовувати більший розмір для Encoder-Decoder
NUM_EPOCHS = 30  # Збільшимо епохи

FEATURE_COLUMNS = [
    "ALLSKY_SFC_SW_DWN", "WS2M", "PS", "RH2M", "T2M",
    "SNODP", "TS", "WD10M", "T2MDEW", "PRECTOTCORR"
]
LABEL_COLUMNS = FEATURE_COLUMNS.copy()


# --- Data Processing Class (без змін) ---
class WeatherDataProcessor:
    def __init__(self, data_directory, sequence_length, forecast_horizon, feature_cols, label_cols):
        self.data_directory = data_directory
        self.sequence_length = sequence_length
        self.forecast_horizon = forecast_horizon
        self.feature_cols = feature_cols
        self.label_cols = label_cols

        logging.info("Loading and parsing CSV data...")
        self.all_data = self._load_and_parse_csv_data()

        self.feature_scaler = MinMaxScaler()
        self.label_scaler = MinMaxScaler()

        logging.info("Fitting scalers...")
        self._fit_scalers()

    def _load_and_parse_csv_data(self):
        all_dfs = []
        for filename in sorted(os.listdir(self.data_directory)):
            if filename.endswith(".csv"):
                filepath = os.path.join(self.data_directory, filename)
                logging.info(f"Parsing file: {filename}")
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

    def get_training_data(self):
        features_scaled = self.feature_scaler.transform(self.all_data[self.feature_cols])
        labels_scaled = self.label_scaler.transform(self.all_data[self.label_cols])

        X, y = [], []
        for i in range(len(self.all_data) - self.sequence_length - self.forecast_horizon + 1):
            X.append(features_scaled[i: i + self.sequence_length])
            y.append(labels_scaled[i + self.sequence_length: i + self.sequence_length + self.forecast_horizon])

        return np.array(X), np.array(y)

    def normalize_features(self, data_df):
        return self.feature_scaler.transform(data_df[self.feature_cols])

    def denormalize_labels(self, scaled_labels):
        return self.label_scaler.inverse_transform(scaled_labels)


# --- Model Definition Function (Оновлена: Encoder-Decoder) ---
def build_encoder_decoder_lstm_forecast(input_shape, output_size, lstm_layer_size, forecast_horizon):
    """
    Будує модель Keras Encoder-Decoder LSTM для прогнозування послідовності.
    input_shape: (sequence_length, feature_size)
    output_size: кількість вихідних міток на кожен часовий крок.
    lstm_layer_size: кількість нейронів в LSTM шарах.
    forecast_horizon: скільки часових кроків вперед прогнозувати.
    """
    encoder_inputs = Input(shape=input_shape)

    # Encoder LSTM: повертає тільки останній стан (контекстний вектор)
    encoder_lstm = LSTM(lstm_layer_size, activation='tanh', return_state=True)
    encoder_outputs, state_h, state_c = encoder_lstm(encoder_inputs)
    encoder_states = [state_h, state_c]  # Зберігаємо стани h та c

    # RepeatVector: повторює контекстний вектор `forecast_horizon` разів
    # Це створює вхідну послідовність для декодера, де кожен елемент - це контекстний вектор
    decoder_inputs = RepeatVector(forecast_horizon)(encoder_outputs)

    # Decoder LSTM: приймає контекстний вектор та генерує вихідну послідовність
    # Передаємо стани енкодера для ініціалізації декодера
    decoder_lstm = LSTM(lstm_layer_size, activation='tanh', return_sequences=True)(decoder_inputs,
                                                                                   initial_state=encoder_states)

    # TimeDistributed Dense шар для прогнозування кожної мітки на кожному часовому кроці декодера
    decoder_outputs = TimeDistributed(Dense(output_size, activation='linear'))(decoder_lstm)

    model = Model(inputs=encoder_inputs, outputs=decoder_outputs)
    model.compile(optimizer=Adam(learning_rate=0.005), loss=MeanSquaredError())
    return model


# --- Main Training Logic ---
if __name__ == "__main__":
    processor = WeatherDataProcessor(DATA_DIRECTORY, SEQUENCE_LENGTH, FORECAST_HORIZON, FEATURE_COLUMNS, LABEL_COLUMNS)
    X_train, y_train = processor.get_training_data()

    logging.info(f"Input shape for model (X_train): {X_train.shape}")
    logging.info(f"Output shape for model (y_train): {y_train.shape}")
    logging.info(f"Total examples for training: {len(X_train)}")

    feature_size = len(FEATURE_COLUMNS)
    label_size = len(LABEL_COLUMNS)
    model = build_encoder_decoder_lstm_forecast(  # Використовуємо нову функцію
        input_shape=(SEQUENCE_LENGTH, feature_size),
        output_size=label_size,
        lstm_layer_size=LSTM_LAYER_SIZE,
        forecast_horizon=FORECAST_HORIZON
    )
    model.summary(print_fn=logging.info)

    logging.info("Starting model training...")
    history = model.fit(
        X_train, y_train,
        epochs=NUM_EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=1
    )
    logging.info("Model training complete.")

    if history:
        plt.figure(figsize=(10, 6))
        plt.plot(history.history['loss'], label='Training Loss')
        plt.title(f'Model Loss Over Epochs (All Params {FORECAST_HORIZON}h Forecast)')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'training_loss_all_params_{FORECAST_HORIZON}h_encdec.png')  # Новий шлях для графіка
        logging.info(f"Training loss plot saved to training_loss_all_params_{FORECAST_HORIZON}h_encdec.png")

    model.save(MODEL_PATH)
    logging.info(f"Model saved to: {MODEL_PATH}")