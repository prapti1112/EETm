"""Transformer model following 'Attention is All We Need' by Vaswani et al."""
import tensorflow as tf
from logzero import logger
from tensorflow.keras import layers


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
    
    # positional encoder implemented using reference from: https://machinelearningmastery.com/a-gentle-introduction-to-positional-encoding-in-transformer-models-part-1/
    def Positional_Encoder(self, x, n=10_000):
        """Generate positional encoding for the given input vector

        Args:
            x (): .
            n (optional): . Defaults to 10_000.

        Returns:
            encoding: positional encoding
        """
        encoding = []
        for record in x:
            record_encoding = []
            for i in range(len(record)):
                angle = record[i]/(n**(2*i/self.d))
                record_encoding.append( np.sin(angle) if i%2==0 else np.cos(angle) )
            encoding.append(record_encoding)

        return np.array(encoding)
    
    def __call__(self, x, h=6):
        positional_encoding = self.Positional_Encoder(x)
        x += positional_encoding

        for encoder_blocker_counter in range(h):
            x += self.mha(self.query(x), self.key(x), self.value(x))
            x = self.layer_norm(x)

            x_fc = layers.Dense(4*self.d, activation="relu")(x)       # set as per formula: 4*d_model
            x_fc = layers.Dense(self.d)(x_fc)
            x += x_fc
            x = self.layer_norm(x)

        return x


class Decoder(tf.keras.Model):
    def __init__(self):
        super(Decoder, self).__init__()


class Transformer(tf.keras.Model):
    def __init__(self):
        super(Transformer, self).__init__()

if __name__ == "__main__":
    import numpy as np

    input_dim = 6
    output_dim = 1
    hidden_dim = 128
    seq_len = 100
    batch_size = 1
    dummpy_input = np.ones(shape=(batch_size, seq_len, input_dim))

    # logger.debug(dummpy_input.shape)
    encoder = Encoder()
    enc_out = encoder(dummpy_input)
    logger.debug(enc_out.shape)