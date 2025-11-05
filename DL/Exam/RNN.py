# ----- Import Libraries -----
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
import matplotlib.pyplot as plt

# ----- Load and Prepare the MNIST Dataset -----
mnist = tf.keras.datasets.mnist            #cifar10 = tf.keras.datasets.cifar10
(x_train, y_train), (x_test, y_test) = mnist.load_data()        #For cifar, cifar10.load_data()

# Normalize the images (values between 0 and 1)
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Treat each image as a sequence of 28 time steps with 28 features each
# So input shape = (28, 28)
# No need to reshape since MNIST is already (num_samples, 28, 28)

#For cifar10 images, resizing should be done
'''x_train = x_train.reshape(-1, 32, 96)
x_test = x_test.reshape(-1, 32, 96)'''

# ----- Build the RNN Model -----
model = Sequential([
    SimpleRNN(128, activation='tanh', input_shape=(28, 28), return_sequences=False),        #For cifar10, (32, 96)
    Dropout(0.3),
    Dense(64, activation='relu'),                #Dense(128)
    Dropout(0.3),
    Dense(10, activation='softmax')  # Output layer for 10 digit classes
])

# ----- Compile the Model -----
model.compile(optimizer=Adam(),
              loss=SparseCategoricalCrossentropy(),
              metrics=['accuracy'])

# ----- Train the Model -----
history = model.fit(x_train, y_train,
                    epochs=5,
                    batch_size=128,
                    validation_data=(x_test, y_test))

# ----- Evaluate the Model -----
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"\n✅ Test Accuracy: {test_acc:.4f}")

# ----- Plot Accuracy and Loss -----
plt.figure(figsize=(12, 5))

# Accuracy plot
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

# Loss plot
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
