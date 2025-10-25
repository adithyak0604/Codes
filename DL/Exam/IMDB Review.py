# ---------------------------------------------
# IMDB Review Classification using RNN (LSTM)
# ---------------------------------------------
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense, LSTM
import matplotlib.pyplot as plt

# ---------------------
# 1. Load the dataset
# ---------------------
# Keep only the top 10,000 most frequent words
num_words = 10000
maxlen = 200   # truncate or pad reviews to 200 words

(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=num_words)

print("Training samples:", len(x_train))
print("Test samples:", len(x_test))

# ---------------------
# 2. Preprocess data
# ---------------------
x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)

# ---------------------
# 3. Build RNN Model
# ---------------------
model = Sequential([
    Embedding(input_dim=num_words, output_dim=128, input_length=maxlen),
    SimpleRNN(64, activation='tanh'),
    #LSTM(64, dropout=0.2, recurrent_dropout=0.2),
    Dense(1, activation='sigmoid')
])

# ---------------------
# 4. Compile the model
# ---------------------
model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)


# ---------------------
# 5. Train the model
# ---------------------
history = model.fit(
    x_train, y_train,
    epochs=5,
    batch_size=64,
    validation_split=0.2,
    verbose=1
)

# ---------------------
# 6. Evaluate the model
# ---------------------
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=1)
print(f"\nTest Accuracy: {test_acc*100:.2f}%")

# ---------------------
# Plot accuracy graph
# ---------------------

# Keras history keys vary; try common names
train_acc = history.history.get('accuracy', history.history.get('acc'))
val_acc = history.history.get('val_accuracy', history.history.get('val_acc'))
train_loss = history.history.get('loss')
val_loss = history.history.get('val_loss')

plt.figure(figsize=(8, 6))
plt.plot(train_acc, marker='o', label='Train Accuracy')
plt.plot(val_acc, marker='o', label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim(0, 1)
plt.grid(True) 
plt.legend()
plt.tight_layout()
plt.savefig('accuracy_plot.png')
plt.show()

plt.figure(figsize=(8, 6))
plt.plot(train_loss, marker='o', label='Train Loss')
plt.plot(val_loss, marker='o', label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig('loss_plot.png')
plt.show()
