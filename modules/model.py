from tensorflow import keras
from tensorflow import config as tf_config

tf_config.experimental.set_memory_growth(
    tf_config.list_physical_devices("GPU")[0], True
)


def new(vocab_size, embidding_dim=256, rnn_units=1024, batch_size=1):
    return keras.Sequential(
        [
            keras.layers.Embedding(
                vocab_size, embidding_dim, batch_input_shape=[batch_size, None]
            ),
            keras.layers.LSTM(
                rnn_units,
                return_sequences=True,
                stateful=True,
                recurrent_initializer="glorot_uniform",
                dropout=0.25,
            ),
            keras.layers.Dense(vocab_size),
        ]
    )
