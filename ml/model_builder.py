import os

os.environ.setdefault("KERAS_BACKEND", "torch")

from keras import Model
from keras.layers import Concatenate, Dense, Input, LSTM

from ml.config import DENSE_UNITS, LOOK_BACK, LSTM_UNITS, N_FUTURE_FEATURES, N_SEQUENCE_FEATURES


def build_lstm_model(
    look_back: int = LOOK_BACK,
    n_sequence_features: int = N_SEQUENCE_FEATURES,
    n_future_features: int = N_FUTURE_FEATURES,
    lstm_units: int = LSTM_UNITS,
    dense_units: int = DENSE_UNITS,
) -> Model:
    """LSTM(32) na sequência + clima futuro concatenado + Dense(16) + Dense(1)."""
    sequence_input = Input(shape=(look_back, n_sequence_features), name="sequence")
    future_input = Input(shape=(n_future_features,), name="future")

    sequence_encoding = LSTM(lstm_units, name="lstm")(sequence_input)
    merged = Concatenate(name="concat")([sequence_encoding, future_input])
    hidden = Dense(dense_units, activation="relu", name="dense")(merged)
    output = Dense(1, name="growth")(hidden)

    model = Model(inputs=[sequence_input, future_input], outputs=output, name="grass_growth_lstm")
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model
