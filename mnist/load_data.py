from tensorflow.keras.datasets import mnist
import numpy as np

class DataLoader:
    def load(self, is_reshape=True, is_one_hot=False):
        (x_train, y_train), (x_test, y_test) = mnist.load_data()

        # convert
        x_train = self._normalize(x_train)
        x_test = self._normalize(x_test)
        if is_reshape:
            x_train = x_train.reshape(x_train.shape[0], -1)
            x_test = x_test.reshape(x_test.shape[0], -1)
        if is_one_hot:
            y_train = self._one_hot(y_train)
            y_test = self._one_hot(y_test)
        
        return (x_train, y_train), (x_test, y_test)

    def _one_hot(self, y_):
        one_hot = np.array([[int(i == y) for i in range(10)] for y in y_], dtype=float)
        return one_hot
    
    def _normalize(self, x_):
        return x_ / 255.0


