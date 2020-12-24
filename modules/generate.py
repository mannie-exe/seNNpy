from sys import exit as sysexit
import tensorflow as tf
import modules.data as data
import modules.model as model


def init():
    seNNpy = model.new(data.vocab_size)
    try:
        seNNpy.load_weights(tf.train.latest_checkpoint(data.CHECKPOINT_DIR))
    except:
        sysexit("Failed to load model checkpoints. Have you trained 💪 yet, seNNpy?")
    seNNpy.build(tf.TensorShape([1, None]))
    return seNNpy


seNNpy = init()


def text(start_str, count=1000, temperature=1.0, model=seNNpy):
    model.reset_states()

    input_eval = [data.char2idx[c] for c in start_str]
    input_eval = tf.expand_dims(input_eval, 0)
    text_generated = []
    for i in range(count):
        predictions = model(input_eval)
        predictions = tf.squeeze(predictions, 0)
        predictions = predictions / temperature
        next_char_idx = tf.random.categorical(predictions, num_samples=1)[-1, 0].numpy()
        input_eval = tf.expand_dims([next_char_idx], 0)
        text_generated.append(data.idx2char[next_char_idx])
    return start_str + "".join(text_generated)
