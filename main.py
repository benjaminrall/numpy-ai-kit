import numpyai
from numpyai.backend import one_hot_encode
from numpyai.nn.layers import Conv2D, MaxPooling2D, Flatten, Dropout, Dense
from numpyai.nn.datasets import MNIST

# Sets data parameters
num_classes = 10
input_shape = (28, 28, 1)

# Loads MNIST sample dataset from remote
(x_train, y_train), (x_test, y_test) = MNIST.load_data()

# Scale images to [0, 1] range
x_train = x_train / 255
x_test = x_test / 255

# Performs one-hot encoding on class labels
y_train = one_hot_encode(y_train, num_classes)
y_test = one_hot_encode(y_test, num_classes)

# Creates the network
network = numpyai.nn.Network([
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D(),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(),
    Flatten(),
    Dropout(0.5),
    Dense(10, activation='softmax')
])

# Builds and compiles the network
network.build(input_shape)
network.compile(
    optimiser='adam', 
    loss='categorical_crossentropy', 
    metrics=['categorical_accuracy']
)

# Prints a summary of the network before training
network.summary()

# Trains the network
batch_size = 128
epochs = 15
network.fit(x_train, y_train, batch_size, epochs, validation_split=0.1)

# Evaluates and saves the trained network
network.evaluate(x_test, y_test)
network.save('mnist_network.pyai')