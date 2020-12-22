from os.path import abspath, join as joinpath
import time
import datetime
import tensorflow as tf
import data
import model


def create_training_chunk(chunk):
    return chunk[:-1], chunk[1:]


@tf.function
def train_step(model, model_input, model_target, optimizer, loss_train_func):
    with tf.GradientTape() as tape:
        predictions = model(model_input)
        loss = tf.reduce_mean(
            tf.keras.losses.sparse_categorical_crossentropy(
                model_target, predictions, from_logits=True
            )
        )
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    loss_train_func(loss)
    return loss


def run(epochs, seq_size=100, batch_size=64, buffer_size=1000):
    # Configure logging and checkpoint storage
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    train_log_dir = abspath("./logs/gradient_tape/" + current_time + "/train")
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    checkpoint_prefix = joinpath(data.CHECKPOINT_DIR, "ckpt_{epoch}")
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_prefix,
        save_weights_only=True,
    )

    seNNpy = model.new(data.vocab_size, batch_size=batch_size)

    # Configure model dataset
    dataset = (
        (
            tf.data.Dataset.from_tensor_slices(data.int_repr)
            .batch(seq_size + 1, drop_remainder=True)
            .map(create_training_chunk)
        )
        .shuffle(buffer_size)
        .batch(batch_size, drop_remainder=True)
    )
    dataset_size = len(dataset)

    optimizer = tf.keras.optimizers.Adam()
    train_loss_func = tf.keras.metrics.Mean("train_loss", dtype=tf.float32)
    for epoch in range(epochs):
        seNNpy.reset_states()

        start = time.time()
        print("Epoch: {} / {}".format(epoch + 1, epochs))
        for (batch_size, (input_text, target_text)) in enumerate(dataset):
            loss = train_step(
                seNNpy, input_text, target_text, optimizer, train_loss_func
            )

            print(
                "\r[{:50s}] {:.1%} ".format(
                    "#" * int((batch_size + 1) / dataset_size * 50),
                    (batch_size + 1) / dataset_size,
                )
                + "loss: {:.4f}".format(loss),
                end="",
                flush=True,
            )

            with train_summary_writer.as_default():
                tf.summary.scalar("loss", train_loss_func.result(), step=epoch)
        time_elapsed = time.time() - start
        print(
            "\nTime taken last epoch {:.3f}s with {:.4f} loss\n".format(
                time_elapsed, loss
            )
        )
        print(
            "Estimated time remaining: {:.3f}s".format(
                time_elapsed * (epochs - (epoch + 1))
            )
        )

        # Save weights every 5th epoch
        if (epoch + 1) % 5 == 0:
            seNNpy.save_weights(checkpoint_prefix.format(epoch=epoch + 1))

        train_loss_func.reset_states()

    seNNpy.save_weights(checkpoint_prefix.format(epoch=epochs))
