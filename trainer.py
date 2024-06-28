import os
import numpy as np
from tensorflow import keras
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical

train_images = np.load('./data/training_logos.npy')
train_labels = np.load('./data/training_labels.npy')

train_images = train_images[:, :, :, :3]

# Encode labels using LabelEncoder
label_encoder = LabelEncoder()
train_labels_encoded = label_encoder.fit_transform(train_labels)

# Convert encoded labels to one-hot encoding
num_classes = len(label_encoder.classes_)
train_labels_one_hot = to_categorical(train_labels_encoded, num_classes=num_classes)

# Define the CNN model
model = keras.Sequential([
    keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation="relu", input_shape=(28, 28, 3)),
    keras.layers.MaxPooling2D(pool_size=(2, 2)),
    keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation="relu"),
    keras.layers.MaxPooling2D(pool_size=(2, 2)),
    keras.layers.Flatten(),
    keras.layers.Dense(256, activation="relu"),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(num_classes, activation="softmax")
])

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(train_images, train_labels_one_hot, epochs=10, batch_size=128)

# Evaluate the trained model
test_loss, test_acc = model.evaluate(train_images, train_labels_one_hot, verbose=2)
print('\nTest accuracy:', test_acc)

# Define the path to save the trained model
model_dir = 'model'
model_filename = 'trained_model.h5'
model_path = os.path.join(model_dir, model_filename)

# Check if the model directory exists, if not, create it
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

# Save the trained model
model.save(model_path)
print("Model saved successfully.")
