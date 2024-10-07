"""Transformer model following 'Attention is All We Need' by Vaswani et al."""
import logging
import tensorflow as tf
from logzero import logger
from tensorflow.keras import layers


# positional encoder implemented using reference from: https://machinelearningmastery.com/a-gentle-introduction-to-positional-encoding-in-transformer-models-part-1/
def Positional_Encoder(x, d, n=10_000):
    """Generate positional encoding for the given input vector

    Args:
        x (): .
        d (): .
        n (optional): . Defaults to 10_000.

    Returns:
        encoding: positional encoding
    """
    encoding = []
    for record in x:
        record_encoding = []
        for i in range(len(record)):
            angle = record[i]/(n**(2*i/d))
            record_encoding.append( np.sin(angle) if i%2==0 else np.cos(angle) )
        encoding.append(record_encoding)

    return np.array(encoding)


class Encoder(tf.keras.Model):
    # Default values are calculated base on the formular: key_dims * num_heads = embedding_dimension
    def __init__(self, mha_num_heads = 2, key_dim=3, seq_len=100):
        super(Encoder, self).__init__()
        self.d = mha_num_heads*key_dim
        
        self.query = layers.Dense(self.d, name="query_weights")
        self.key = layers.Dense(self.d, name="key_weights")
        self.value =layers.Dense(self.d, name="value_weights")
        self.mha = layers.MultiHeadAttention(num_heads = mha_num_heads, key_dim=key_dim)

        self.layer_norm = layers.LayerNormalization(axis=-1, epsilon=1e-6)
    
    def __call__(self, x, h=6):
        positional_encoding = Positional_Encoder(x, self.d)
        x += positional_encoding

        for encoder_blocker_counter in range(h):
            x += self.mha(query=self.query(x), key=self.key(x), value=self.value(x))
            x = self.layer_norm(x)

            x_fc = layers.Dense(4*self.d, activation="relu")(x)       # set as per formula: 4*d_model
            x_fc = layers.Dense(self.d)(x_fc)
            x += x_fc
            x = self.layer_norm(x)

        return x


class Decoder(tf.keras.Model):
    def __init__(self, mha_num_heads = 2, key_dim=3):
        super(Decoder, self).__init__()
        self.d = mha_num_heads * key_dim

        self.query = layers.Dense(self.d, name="query_weights")
        self.key = layers.Dense(self.d, name="key_weights")
        self.value =layers.Dense(self.d, name="value_weights")
        self.mha1 = layers.MultiHeadAttention(num_heads = mha_num_heads, key_dim=key_dim)
        self.mha2 = layers.MultiHeadAttention(num_heads = mha_num_heads, key_dim=key_dim)

        self.layer_norm = layers.LayerNormalization(axis=-1, epsilon=1e-6)
    
    def __call__(self, x):
        dec_out_prev = x.numpy().reshape((-1, x.shape[-1]))
        dec_out_prev[:-1] = dec_out_prev[1:]
        dec_out_prev[0, :] = 0
        dec_out_prev = tf.convert_to_tensor(dec_out_prev.reshape(x.shape))

        positional_encoding = Positional_Encoder(x, self.d)
        dec_out_prev += positional_encoding

        dec_out_prev += self.mha1(query=self.query(dec_out_prev), key=self.key(dec_out_prev), value=self.value(dec_out_prev))
        dec_out_prev = self.layer_norm(dec_out_prev)

        dec_out_prev += self.mha2(query=self.query(dec_out_prev), key=self.key(x), value=self.value(x), use_causal_mask=True)
        x = self.layer_norm(dec_out_prev)

        x_fc = layers.Dense(4*self.d, activation="relu")(x)       # set as per formula: 4*d_model
        x_fc = layers.Dense(self.d)(x_fc)

        return x_fc


class Transformer(tf.keras.Model): 
    def __init__(self, encoder, decoder):
        super(Transformer, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
    
    def __call__(self, x):
        enc_out = self.encoder(x)
        dec_out = self.decoder(enc_out)

        final_output = layers.Dense(1)(dec_out)
        return final_output


if __name__ == "__main__":
    import numpy as np

    input_dim = 6
    output_dim = 1
    hidden_dim = 128
    seq_len = 100
    batch_size = 1
    dummpy_input = np.random.randn(batch_size, seq_len, input_dim)
    logger.debug(dummpy_input.shape)

    encoder = Encoder()
    decoder = Decoder()

    model = Transformer(encoder, decoder)
    out = model(dummpy_input)
    logger.debug(out.shape)