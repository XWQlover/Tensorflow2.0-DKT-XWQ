import tensorflow as tf


class DKT(tf.keras.models.Model):
    def __init__(self, total_skills_correctness, embedding_size):
        super(DKT, self).__init__(name="DKTModel")

        self.mask = tf.keras.layers.Masking(mask_value=-1)

        # 两个嵌入层

        self.skill_embedding = tf.keras.layers.Embedding(total_skills_correctness, embedding_size)
        # RNN
        self.rnn = tf.keras.Sequential([tf.keras.layers.LSTM(units=64, return_sequences=True, dropout=0.3),
                                        tf.keras.layers.LSTM(units=64, return_sequences=True, dropout=0.3)])

        # dense
        self.dense = tf.keras.layers.Dense(total_skills_correctness / 2, activation='sigmoid')

        self.distribute = tf.keras.layers.TimeDistributed(self.dense)

        self.softmax = tf.keras.layers.Softmax()

    def call(self, skillid):
        skillid = tf.expand_dims(skillid, axis=-1)

        skillid = self.mask(skillid)

        skill_vector = self.skill_embedding(skillid)

        x = skill_vector

        x = tf.squeeze(x, axis=-2)
        x = self.rnn(x)
        y = self.distribute(x)

        return y