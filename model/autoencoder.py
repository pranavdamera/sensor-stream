"""LSTM autoencoder architecture for multivariate time-series reconstruction."""

from __future__ import annotations

from tensorflow import keras
from tensorflow.keras import layers


def build_lstm_autoencoder(
    window_size: int,
    n_features: int,
) -> keras.Model:
    """Build and compile an LSTM autoencoder (not trained in this module).

    Architecture: encoder LSTM stacks, RepeatVector bottleneck, decoder LSTM
    stacks with a time-distributed output projection. Loss is mean squared
    error; optimizer is Adam with learning rate ``1e-3``.

    Args:
        window_size: Number of timesteps per window (sequence length).
        n_features: Number of input features per timestep.

    Returns:
        A compiled ``keras.Model`` ready for ``fit``.
    """
    inputs = keras.Input(shape=(window_size, n_features))

    x = layers.LSTM(64, return_sequences=True)(inputs)
    x = layers.LSTM(32, return_sequences=False)(x)
    x = layers.RepeatVector(window_size)(x)
    x = layers.LSTM(32, return_sequences=True)(x)
    x = layers.LSTM(64, return_sequences=True)(x)
    outputs = layers.TimeDistributed(layers.Dense(n_features))(x)

    model = keras.Model(inputs, outputs, name="lstm_autoencoder")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="mse",
    )
    return model
