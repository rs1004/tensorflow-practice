from load_data import DataLoader
from network import SimpleNet

data_loader = DataLoader()
(x_train, y_train), (x_test, y_test) = data_loader.load()

model = SimpleNet()._create_model()

model.fit(x_train, y_train, epochs=10)
test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)

print(f'test acc: {test_accuracy}')

