from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential

class SimpleNet:
    def _create_model(self):
        model = Sequential([
            Dense(units=500, activation='relu', input_shape=(28 * 28,)),
            Dropout(0.2),
            Dense(units=10, activation='softmax')
        ])

        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['sparse_categorical_accuracy'])

        return model
    
