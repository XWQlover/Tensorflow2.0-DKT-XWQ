from DKTModel import DKT
import tensorflow as tf
from dataUtils import AssismentData

ass = AssismentData()
train_data,test_data,val_data = ass.datasetReturn()
val_log = 'log/val'
train_loss_log = 'log/train'
summary_writer = tf.summary.create_file_writer(val_log)

dkt = DKT(ass.data["user_id"].max() + 1, ass.data["sequence_id"].max() + 1, 64)
AUC = tf.keras.metrics.AUC()
VAUC = tf.keras.metrics.AUC()
SCC = tf.keras.metrics.SparseCategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adam(learning_rate=0.005)


def test_one_step(userid, skillid, label):
    loss = dkt(userid, skillid)

    label = tf.expand_dims(label, axis=-1)

    mask = 1. - tf.cast(tf.equal(label, -1.), label.dtype)
    mask = tf.squeeze(mask)
    label = tf.boolean_mask(label, mask=mask)
    loss = tf.boolean_mask(loss, mask=mask)

    VAUC.update_state(tf.one_hot(tf.squeeze(tf.cast(label, tf.int32), axis=-1), depth=2), loss)


def train_one_step(userid, skillid, label):
    with tf.GradientTape() as tape:
        loss = dkt(userid, skillid)

        label = tf.expand_dims(label, axis=-1)

        mask = 1. - tf.cast(tf.equal(label, -1.), label.dtype)
        mask = tf.squeeze(mask)

        label = tf.boolean_mask(label, mask=mask)
        loss = tf.boolean_mask(loss, mask=mask)

        AUC.update_state(tf.one_hot(tf.squeeze(tf.cast(label, tf.int32), axis=-1), depth=2), loss)
        SCC.update_state(label, loss)

        loss = tf.keras.losses.sparse_categorical_crossentropy(label, loss)

        loss = tf.reduce_sum(loss)

        gradients = tape.gradient(loss, dkt.trainable_variables)
        # 反向传播，自动微分计算
        optimizer.apply_gradients(zip(gradients, dkt.trainable_variables))

for epoch in range(10):
  train_data = train_data.shuffle(32)
  AUC.reset_states()
  VAUC.reset_states()
  SCC.reset_states()
  for u,v,l in train_data.as_numpy_iterator():
    train_one_step(u,v,l)

  for u,v,l in val_data.as_numpy_iterator():
    test_one_step(u,v,l)

  with summary_writer.as_default():
    tf.summary.scalar('train_auc',AUC.result(),step=epoch)
    tf.summary.scalar('val_auc',VAUC.result(),step=epoch)
  print(SCC.result(),AUC.result(),VAUC.result())