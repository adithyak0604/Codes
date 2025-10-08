import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import SGD, Adam
import matplotlib.pyplot as plt
import numpy as np

# 1. Load and preprocess data
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# 2. Define a function to create the model
def create_mlp_model():
    model = Sequential([
        Flatten(input_shape=(28, 28)),
        Dense(128, activation='relu'),
        Dense(128, activation='relu'),
        Dense(10, activation='softmax')
    ])
    return model

# 3. Define the optimizers to test
optimizers_to_test = {
    'SGD': SGD(),
    'SGD with Momentum': SGD(momentum=0.9),
    'Adam': Adam()
}

histories = {}
epochs = 15

# 4. Loop, compile, and train a model for each optimizer
for name, optimizer in optimizers_to_test.items():
    print(f"\n--- Training with {name} optimizer ---")
    model = create_mlp_model()
    model.compile(optimizer=optimizer,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    history = model.fit(x_train, y_train,
                        validation_data=(x_test, y_test),
                        epochs=epochs,
                        batch_size=64,
                        verbose=0) # Set to 1 to see live progress
    histories[name] = history.history
    print(f"Final Validation Accuracy ({name}): {history.history['val_accuracy'][-1]:.4f}")

# 5. Plot the results
plt.style.use('seaborn-v0_8-whitegrid')

# Plot Accuracy
plt.figure(figsize=(12, 8))
for name, history in histories.items():
    plt.plot(history['val_accuracy'], label=f'{name} Validation Accuracy')
plt.title('Validation Accuracy Comparison for Optimizers', fontsize=16)
plt.xlabel('Epochs', fontsize=12)
plt.ylabel('Accuracy', fontsize=12)
plt.legend(fontsize=12)
plt.grid(True)
plt.savefig('optimizer_accuracy_comparison.png')

# Plot Loss
plt.figure(figsize=(12, 8))
for name, history in histories.items():
    plt.plot(history['val_loss'], label=f'{name} Validation Loss')
plt.title('Validation Loss Comparison for Optimizers', fontsize=16)
plt.xlabel('Epochs', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.legend(fontsize=12)
plt.grid(True)
plt.savefig('optimizer_loss_comparison.png')

print("\nPlots saved as 'optimizer_accuracy_comparison.png' and 'optimizer_loss_comparison.png'")
