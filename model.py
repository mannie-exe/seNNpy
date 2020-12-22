from tensorflow import keras


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
                dropout=0.5,
            ),
            keras.layers.Dense(vocab_size),
        ]
    )
