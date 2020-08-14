import tensorflow as tf

class DKT(tf.keras.models.Model):
    def __init__(self, total_user, total_skill, embedding_size):
        super(DKT, self).__init__(name="DKTModel")

        self.mask = tf.keras.layers.Masking(mask_value=-1.0)

        # 两个嵌入层
        self.user_embedding = tf.keras.layers.Embedding(total_user, embedding_size)
        self.skill_embedding = tf.keras.layers.Embedding(total_skill, embedding_size)
        # RNN
        self.rnn = tf.keras.layers.LSTM(units=64, return_sequences=True, dropout=0.3)

        # dense
        self.dense = tf.keras.layers.Dense(2, activation='sigmoid')

        self.distribute = tf.keras.layers.TimeDistributed(self.dense)

        self.softmax = tf.keras.layers.Softmax()

    def call(self, userid, skillid):
        userid, skillid = tf.expand_dims(userid, axis=-1), tf.expand_dims(skillid, axis=-1)
        userid = self.mask(userid)
        skillid = self.mask(skillid)

        user_vector = self.user_embedding(userid)

        skill_vector = self.skill_embedding(skillid)

        x = tf.concat([user_vector, skill_vector], axis=-1)

        x = tf.squeeze(x, axis=-2)
        x = self.rnn(x)

        x = self.distribute(x)

        y = self.softmax(x)

        return y