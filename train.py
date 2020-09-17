from DKTModel import DKT
import tensorflow as tf
from dataUtils import AssismentData

dkt = DKT(int(ass.data["skills_correctness"].max() + 1), 32)
skill_num = int((ass.data["skills_correctness"].max() + 1) / 2)
print(skill_num)
AUC = tf.keras.metrics.AUC()
VAUC = tf.keras.metrics.AUC()
SCC = tf.keras.metrics.BinaryCrossentropy()
optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)


def test_one_step(sequence_id, skillid, label):
    loss = dkt(skillid)

    label = tf.expand_dims(label, axis=-1)

    mask = 1. - tf.cast(tf.equal(label, -1), tf.float32)

    mask = tf.squeeze(mask)

    squenceid = tf.boolean_mask(sequence_id, mask=mask)
    squenceid = tf.one_hot(squenceid, depth=skill_num, axis=-1)
    label = tf.boolean_mask(label, mask=mask)
    loss = tf.boolean_mask(loss, mask=mask)

    loss = tf.expand_dims(tf.reduce_sum(tf.multiply(squenceid, loss), axis=-1), axis=-1)

    loss = tf.expand_dims(loss, axis=-1)

    label = tf.squeeze(tf.one_hot(label, depth=2), axis=1)

    loss = tf.concat([1 - loss, loss], axis=1)

    VAUC.update_state(label, tf.squeeze(loss, axis=-1))


def train_one_step(sequence_id, skillid, label):
    with tf.GradientTape() as tape:
        loss = dkt(skillid)

        label = tf.expand_dims(label, axis=-1)

        mask = 1. - tf.cast(tf.equal(label, -1), tf.float32)
        mask = tf.squeeze(mask)

        sequenceid = tf.boolean_mask(sequence_id, mask=mask)
        sequenceid = tf.one_hot(sequenceid, depth=skill_num, axis=-1)
        label = tf.boolean_mask(label, mask=mask)
        loss = tf.boolean_mask(loss, mask=mask)

        loss = tf.expand_dims(tf.reduce_sum(tf.multiply(sequenceid, loss), axis=-1), axis=-1)

        loss_real = tf.reduce_sum(tf.keras.losses.binary_crossentropy(label, loss))

        SCC.update_state(label, loss)

        loss = tf.expand_dims(loss, axis=-1)

        loss = tf.concat([1 - loss, loss], axis=1)

        label = tf.squeeze(tf.one_hot(label, depth=2), axis=1)

        AUC.update_state(label, tf.squeeze(loss, axis=-1))

        gradients = tape.gradient(loss_real, dkt.trainable_variables)
        # 反向传播，自动微分计算
        optimizer.apply_gradients(zip(gradients, dkt.trainable_variables))


for epoch in range(8):
    train_data = train_data.shuffle(50)
    AUC.reset_states()
    VAUC.reset_states()
    SCC.reset_states()
    for s, p, l in train_data.as_numpy_iterator():
        train_one_step(p, s, l)

    for s, p, l in test_data.as_numpy_iterator():
        test_one_step(p, s, l)

    with summary_writer.as_default():
        tf.summary.scalar('train_auc', AUC.result(), step=epoch)
        tf.summary.scalar('val_auc', VAUC.result(), step=epoch)

    print(SCC.result(), AUC.result(), VAUC.result())