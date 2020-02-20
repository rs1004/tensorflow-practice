import os
from load_data import DataLoader
from network import SimpleNet
from tensorflow.keras.callbacks import ModelCheckpoint
import tensorflow as tf
import re

checkpoint_path = "result_SimpleNet/cp-{epoch:04d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

# チェックポイントコールバックを作る
cp_callback = ModelCheckpoint(checkpoint_path,
                              save_weights_only=True,
                              verbose=1,
                              period=5)

# 設定
initial_epoch = 0
epochs = 10

# 学習・評価
data_loader = DataLoader()
(x_train, y_train), (x_test, y_test) = data_loader.load()

model = SimpleNet()._create_model()
if os.path.exists(checkpoint_dir):
    latest = tf.train.latest_checkpoint(checkpoint_dir)
    model.load_weights(latest)
    initial_epoch = int(re.findall(r'\d{4}', latest)[0])

model.fit(x_train, y_train, initial_epoch=initial_epoch, epochs=initial_epoch + epochs, callbacks=[cp_callback])
test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)

print(f'test acc: {test_accuracy}')
