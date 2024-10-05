"""File containing a basic encoder decoder model"""

import tensorflow as tf
from tensorflow.keras import layers
from logzero import logger


class Encoder(tf.keras.Model):
    def __init__(self, hidden_dim):
        """.

        Args:
            input_dim (int): .
            hidden_dim (int): .
        """
        super(Encoder, self).__init__()
        self.embedding = layers.Dense(hidden_dim)
        self.lstm = layers.LSTM(hidden_dim, return_sequences=True, return_state=True)

    def call(self, x):
        """.

        Args:
            x (np.ndarray): Shape: [batch_size, seq_length, input_dim]

        Returns:
            states(tuple): 
        """
        x = self.embedding(x)  # 
        output, hidden_state, cell_state = self.lstm(x)
        return output, hidden_state, cell_state

class Decoder(tf.keras.Model):
    def __init__(self, output_dim, hidden_dim):
        super(Decoder, self).__init__()
        self.embedding = layers.Dense(hidden_dim)  # Start token dimension is 1
        self.lstm = layers.LSTM(hidden_dim, return_sequences=True, return_state=True)
        self.fc_out = layers.Dense(output_dim)

    def call(self, x, hidden_state, cell_state):
        x = self.embedding(tf.expand_dims(x, axis=-1))  # Shape: (batch_size, 1) -> (batch_size, 1, hidden_dim)
        output_seq, hidden_state, cell_state = self.lstm(x, initial_state=[hidden_state, cell_state])
        output_seq = self.fc_out(output_seq)  # Shape: (batch_size, seq_length=100, output_dim)
        return output_seq

class Seq2Seq(tf.keras.Model):
    def __init__(self, encoder, decoder, batch_size):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.batch_size = batch_size

    def call(self, x):
        """.

        Args:
            x (tf.tensor): Shape - [batch_size, 100, input_dim]

        Returns:
            _type_: _description_
        """
        encoder = Encoder(hidden_dim=hidden_dim)
        _, hidden_state, cell_state = encoder(x)
        
        input_token = tf.zeros((batch_size, 1))  # Start token of dimension 1
        decoder = Decoder(output_dim, hidden_dim)
        final_output = tf.reshape(decoder(input_token, hidden_state, cell_state), shape=(batch_size,))
        
        return final_output


# if __name__ == "__main__":
#     import numpy as np

#     input_dim = 6
#     output_dim = 1
#     hidden_dim = 128
#     seq_len = 100
#     batch_size = 1
#     dummpy_input = np.ones(shape=(batch_size, seq_len, input_dim))

#     encoder = Encoder(hidden_dim=hidden_dim)
#     # enc_out, hidden_state, cell_state = encoder( np.ones(shape=(batch_size, seq_len, input_dim)) )
#     # logger.debug(f"Encoder output shape - Output: {enc_out.shape}, Hidden State: {hidden_state.shape}, Cell State: {cell_state.shape}")
    
#     # input_token = tf.zeros((batch_size, 1))  # Start token of dimension 1
#     # logger.debug(f"Input tokens: {input_token.shape}")
#     decoder = Decoder(output_dim, hidden_dim)
#     # dec_out = decoder(input_token, hidden_state, cell_state)
#     # dec_out = tf.reshape(dec_out, shape=(batch_size,))
#     # logger.debug(f"Decoder output shape: {dec_out.shape}, {dec_out}")

#     model = Seq2Seq(encoder, decoder, batch_size)
#     out = model(dummpy_input)
#     logger.debug(out)



