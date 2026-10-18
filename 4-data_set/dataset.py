import os
import shutil
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import json
import numpy as np

# Function to load and compile the original trained model
def load_and_compile_model(filepath):
    model = tf.keras.models.load_model(filepath)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    # Load the saved history
    with open('original_neural_network_history.json', 'r') as f:
        history = json.load(f)
    model.history = type('History', (object,), {'history': history})()
    
    return model

# Setup data function for MNIST dataset with specific size
def setup_mnist_data(sample_size_per_class=320):
    (x_train_full, y_train_full), (x_test_full, y_test_full) = tf.keras.datasets.mnist.load_data()

    # Normalize the images to [0, 1] range
    x_train_full = x_train_full / 255.0
    x_test_full = x_test_full / 255.0

    # Reshape the images to include the channel dimension (grayscale)
    x_train_full = x_train_full.reshape((-1, 28, 28, 1))
    x_test_full = x_test_full.reshape((-1, 28, 28, 1))

    # Function to sample specific size of images per class
    def sample_images_and_labels(images, labels, sample_size):
        sampled_images = []
        sampled_labels = []
        for i in range(10):  # 10 classes
            idx = np.where(labels == i)[0]
            sampled_idx = np.random.choice(idx, sample_size, replace=False)
            sampled_images.append(images[sampled_idx])
            sampled_labels.append(labels[sampled_idx])
        return np.vstack(sampled_images), np.hstack(sampled_labels)

    # Sample the specific size per class
    x_train, y_train = sample_images_and_labels(x_train_full, y_train_full, sample_size_per_class)
    x_test, y_test = sample_images_and_labels(x_test_full, y_test_full, sample_size_per_class // 4)

    # One-hot encode the labels
    y_train = tf.keras.utils.to_categorical(y_train, 10)
    y_test = tf.keras.utils.to_categorical(y_test, 10)
    
    return (x_train, y_train), (x_test, y_test)

# Function to build the baseline model architecture
def build_baseline_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=(28, 28, 1)),  # Input layer for 28x28 grayscale images
        tf.keras.layers.Conv2D(20, (5, 5), activation='tanh'),  # First convolutional layer
        tf.keras.layers.MaxPooling2D((2, 2)),  # First max pooling layer
        tf.keras.layers.Conv2D(50, (5, 5), activation='tanh'),  # Second convolutional layer
        tf.keras.layers.MaxPooling2D((2, 2)),  # Second max pooling layer
        tf.keras.layers.Flatten(),  # Flatten the 2D output to 1D
        tf.keras.layers.Dense(500, activation='tanh'),  # Fully connected dense layer
        tf.keras.layers.Dense(10, activation='softmax')  # Output layer with 10 classes (0-9 digits)
    ])
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Function to train models and plot results
def train_and_plot_models():
    # Setup data for the baseline model
    def setup_data():
        base_dir = 'Edited_Dataset'
        train_dir = os.path.join(base_dir, 'all_training')
        test_dir = os.path.join(base_dir, 'all_test')

        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(test_dir, exist_ok=True)

        digit_folders = ['0-zero', '1-one', '2-two', '3-three', '4-four', '5-five', '6-six', '7-seven', '8-eight', '9-nine']
        for i, folder in enumerate(digit_folders):
            train_src = os.path.join(base_dir, folder, f'{i}-training', f'digit_{i}')
            test_src = os.path.join(base_dir, folder, f'{i}-test', f'digit_{i}')
            
            train_dst = os.path.join(train_dir, f'digit_{i}')
            test_dst = os.path.join(test_dir, f'digit_{i}')
            
            os.makedirs(train_dst, exist_ok=True)
            os.makedirs(test_dst, exist_ok=True)
            
            for filename in os.listdir(train_src):
                if filename.endswith(".png"):
                    shutil.move(os.path.join(train_src, filename), train_dst)
            
            for filename in os.listdir(test_src):
                if filename.endswith(".png"):
                    shutil.move(os.path.join(test_src, filename), test_dst)

        train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
        test_datagen = ImageDataGenerator(rescale=1./255)
        
        train_generator = train_datagen.flow_from_directory(
            train_dir, target_size=(28, 28), color_mode='grayscale', batch_size=32, class_mode='categorical'
        )
        
        test_generator = test_datagen.flow_from_directory(
            test_dir, target_size=(28, 28), color_mode='grayscale', batch_size=32, class_mode='categorical'
        )

        return train_generator, test_generator

    # Setup data for the MNIST model
    (x_train_mnist, y_train_mnist), (x_test_mnist, y_test_mnist) = setup_mnist_data()

    # Load, compile, and possibly train the original model
    original_model = load_and_compile_model('original_neural_network.h5')

    # Get the original model's history
    original_history = original_model.history.history

    # Build MNIST model
    mnist_model = build_baseline_model()

    # Train MNIST model and store history
    history_mnist = mnist_model.fit(
        x_train_mnist, y_train_mnist,
        epochs=10,
        batch_size=32,
        validation_data=(x_test_mnist, y_test_mnist)
    )

    # Plotting accuracy
    epochs = range(1, 11)

    plt.plot(epochs, original_history['accuracy'], 'bo-', label='Baseline Training accuracy')
    plt.plot(epochs, original_history['val_accuracy'], 'b-', label='Baseline Validation accuracy')

    plt.plot(epochs, history_mnist.history['accuracy'], 'ro-', label='MNIST Training accuracy')
    plt.plot(epochs, history_mnist.history['val_accuracy'], 'r-', label='MNIST Validation accuracy')

    plt.title('Training and Validation Accuracy Comparison')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

# Run the function to train and plot models
train_and_plot_models()
