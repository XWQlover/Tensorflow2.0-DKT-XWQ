import pandas as pd
import numpy as np
import tensorflow as tf


class AssismentData():
    def __init__(self):
        self.data = pd.read_csv("/content/drive/My Drive/DKT/skill_builder_data.csv")
        self.data = self.data.dropna()
        self.data["user_id"], _ = pd.factorize(self.data["user_id"])
        self.data["skill_id"], _ = pd.factorize(self.data["skill_id"])
        self.data["skills_correctness"] = self.data.apply(
            lambda x: x.skill_id * 2 if x.correct == 0.0 else x.skill_id * 2 + 1, axis=1)
        self.data = self.data.groupby("user_id").filter(lambda q: len(q) > 1).copy()
        self.seq = self.data.groupby('user_id').apply(
            lambda r: (
                r["skills_correctness"].values[:-1],
                r["skill_id"].values[1:],
                r['correct'].values[1:]
            )
        )

    def datasetReturn(self, shuffle=None, batch_size=32, val_data=None):

        dataset = tf.data.Dataset.from_generator(lambda: self.seq, output_types=(tf.int32, tf.int32, tf.int32))

        if shuffle:
            dataset = dataset.shuffle(buffer_size=shuffle)

        MASK_VALUE = -1
        dataset = dataset.padded_batch(
            batch_size=50,
            padding_values=(MASK_VALUE, MASK_VALUE, MASK_VALUE),
            padded_shapes=([None], [None], [None]),
            drop_remainder=True
        )
        i = 0
        for l in dataset.as_numpy_iterator():
            i += 1

        dataset = dataset.shuffle(buffer_size=50)
        test_size = int(np.ceil(i * 0.2))
        train_size = i - test_size

        train_data = dataset.take(train_size)
        dataset = dataset.skip(train_size)

        return train_data, dataset