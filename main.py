import tensorflow as tf

import numpy as np
import os
import time
import datetime


# Load and vectorize dataset
# -----------------------------------------------
path_to_file = "./training_data/messages.txt"
text = open(path_to_file, "rb").read().decode(encoding="utf-8")
vocab = sorted(set(text))

char2idx = {u: i for i, u in enumerate(vocab)}
idx2char = np.array(vocab)
text_as_int = np.array([char2idx[c] for c in text])
# -----------------------------------------------


# Batch and create TensorFlow training batches
# -----------------------------------------------
def split_input_target(chunk):
    input_text = chunk[:-1]
    target_text = chunk[1:]
    return input_text, target_text


seq_length = 100

char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)
sequences = char_dataset.batch(seq_length + 1, drop_remainder=True)
dataset = sequences.map(split_input_target)

BATCH_SIZE = 64
BUFFER_SIZE = 1000

dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)
data_size = len(dataset)
# -----------------------------------------------


# Build model and loss function
# -----------------------------------------------
def build_model(vocab_size, embedding_dim, rnn_units, batch_size):
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Embedding(
                vocab_size, embedding_dim, batch_input_shape=[batch_size, None]
            ),
            tf.keras.layers.LSTM(
                rnn_units,
                return_sequences=True,
                stateful=True,
                recurrent_initializer="glorot_uniform",
                dropout=0.5,
            ),
            tf.keras.layers.Dense(vocab_size),
        ]
    )
    return model


def loss(labels, logits):
    return tf.keras.losses.sparse_categorical_crossentropy(
        labels, logits, from_logits=True
    )


vocab_size = len(vocab)
embedding_dim = 256
rnn_units = 1024

model = build_model(
    vocab_size=vocab_size,
    embedding_dim=embedding_dim,
    rnn_units=rnn_units,
    batch_size=BATCH_SIZE,
)
# -----------------------------------------------


# Train model
# -----------------------------------------------
EPOCHS = 30

checkpoint_dir = "./training_checkpoints"
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")

checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_prefix, save_weights_only=True
)

optimizer = tf.keras.optimizers.Adam()

train_loss = tf.keras.metrics.Mean("train_loss", dtype=tf.float32)

current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
train_log_dir = "logs/gradient_tape/" + current_time + "/train"
train_summary_writer = tf.summary.create_file_writer(train_log_dir)


@tf.function
def train_step(inp, target):
    with tf.GradientTape() as tape:
        predictions = model(inp)
        loss = tf.reduce_mean(
            tf.keras.losses.sparse_categorical_crossentropy(
                target, predictions, from_logits=True
            )
        )
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    train_loss(loss)
    return loss


for epoch in range(EPOCHS):
    start = time.time()
    model.reset_states()

    print("Epoch: {}/{}".format(epoch + 1, EPOCHS))
    for (batch_n, (inp, target)) in enumerate(dataset):
        loss = train_step(inp, target)

        print(
            "\r[{:50s}] {:.1f}% ".format(
                "#" * int((batch_n + 1) / data_size * 50),
                ((batch_n + 1) / data_size) * 100,
            )
            + "loss: {:.4f}".format(loss),
            end="",
            flush=True,
        )

        with train_summary_writer.as_default():
            tf.summary.scalar("loss", train_loss.result(), step=epoch)

    if (epoch + 1) % 5 == 0:
        model.save_weights(checkpoint_prefix.format(epoch=epoch))

    d_time = time.time() - start
    print("\nTime taken for 1 epoch {:.3f} sec with {:.4f} loss\n".format(d_time, loss))
    print(
        "Estimated time remaining: {:.3f} sec.".format(d_time * (EPOCHS - (epoch + 1)))
    )

    train_loss.reset_states()

model.save_weights(checkpoint_prefix.format(epoch=EPOCHS))

# -----------------------------------------------


# Re-build model to generate new text
# -----------------------------------------------
def generate_text(model, start_string):
    num_generate = 1000
    temperature = 1.0

    input_eval = [char2idx[s] for s in start_string]
    input_eval = tf.expand_dims(input_eval, 0)

    model.reset_states()

    text_generated = []
    for i in range(num_generate):
        predictions = model(input_eval)
        predictions = tf.squeeze(predictions, 0)
        predictions = predictions / temperature
        predicted_id = tf.random.categorical(predictions, num_samples=1)[-1, 0].numpy()
        input_eval = tf.expand_dims([predicted_id], 0)

        text_generated.append(idx2char[predicted_id])

    return start_string + "".join(text_generated)


model = build_model(vocab_size, embedding_dim, rnn_units, batch_size=1)
model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))
model.build(tf.TensorShape([1, None]))

START_STRING = u"@mannie.exe "

print(generate_text(model, start_string=START_STRING))
# -----------------------------------------------
