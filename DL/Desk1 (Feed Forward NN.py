import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.metrics import SparseCategoricalAccuracy
from tensorflow.keras.optimizers import Adam, RMSprop, SGD

# Load and normalize CIFAR-10 data,, u can also use mnist (better accuracy)
cifar10 = tf.keras.datasets.cifar10 #change cifar10 to mnist if mnist
(x_train, y_train), (x_test, y_test) = cifar10.load_data() #mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# List of optimizer classes
optimizers = [Adam, RMSprop, SGD] #add more optimizers if u want

# Loop through each optimizer
for opt_class in optimizers:
    print(f"\nUsing optimizer: {opt_class.__name__}")

    # Build a fresh model for each optimizer
    model = Sequential([
        Flatten(input_shape=(32, 32, 3)), #(28,28) input shape for mnist
        Dense(128, activation='relu'),
        Dense(64, activation='relu'),
        Dense(10, activation='softmax')
    ])

    # Compile the model with the current optimizer
    model.compile(
        optimizer=opt_class(),
        loss=SparseCategoricalCrossentropy(),
        metrics=[SparseCategoricalAccuracy()]
    )

    # Train the model
    model.fit(x_train, y_train, epochs=5, verbose=1) #change epochs if u want

    # Evaluate the model
    test_loss, test_acc = model.evaluate(x_test, y_test)
    print(f"Test accuracy with {opt_class.__name__}: {test_acc:.4f}")
