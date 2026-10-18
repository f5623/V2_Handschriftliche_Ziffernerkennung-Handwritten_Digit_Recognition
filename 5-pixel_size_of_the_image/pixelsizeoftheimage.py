import os
import shutil
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import json
import time

# Function to load and compile the original trained model
def load_and_compile_model(filepath):
    model = tf.keras.models.load_model(filepath)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    # Load the saved history
    with open('original_neural_network_history.json', 'r') as f:
        history = json.load(f)
    model.history = type('History', (object,), {'history': history})()
    
    return model

# Function to consolidate images into single training and test directories with subfolders for each digit class
def consolidate_images(base_dir='Edited_Dataset'):
    train_dir = os.path.join(base_dir, 'all_training')
    test_dir = os.path.join(base_dir, 'all_test')

    # Create combined directories if they don't exist
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # Function to organize files into subdirectories
    digit_folders = ['0-zero', '1-one', '2-two', '3-three', '4-four', '5-five', '6-six', '7-seven', '8-eight', '9-nine']
    for i, folder in enumerate(digit_folders):
        train_src = os.path.join(base_dir, folder, f'{i}-training', f'digit_{i}')
        test_src = os.path.join(base_dir, folder, f'{i}-test', f'digit_{i}')
        
        train_dst = os.path.join(train_dir, f'digit_{i}')
        test_dst = os.path.join(test_dir, f'digit_{i}')
        
        os.makedirs(train_dst, exist_ok=True)  # Ensure training subdirectory exists
        os.makedirs(test_dst, exist_ok=True)  # Ensure test subdirectory exists
        
        for filename in os.listdir(train_src):
            if filename.endswith(".png"):
                shutil.move(os.path.join(train_src, filename), train_dst)  # Move training images to the subdirectory
        
        for filename in os.listdir(test_src):
            if filename.endswith(".png"):
                shutil.move(os.path.join(test_src, filename), test_dst)  # Move test images to the subdirectory

    return train_dir, test_dir

# Function to build the baseline model architecture
def build_baseline_model(input_shape):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=input_shape),  # Input layer
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

# Function to train and evaluate models with different pixel sizes
def train_and_evaluate_model(target_size, input_shape):
    # Consolidate images
    train_dir, test_dir = consolidate_images()

    # Data augmentation and normalization for training
    train_datagen = ImageDataGenerator(
        rescale=1./255,  # Rescale images from [0,255] to [0,1]
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True
    )
    test_datagen = ImageDataGenerator(rescale=1./255)  # Only rescale for test data

    # Load data from combined directories
    train_generator = train_datagen.flow_from_directory(
        train_dir,  # Path to training data
        target_size=target_size,
        color_mode='grayscale',
        batch_size=32,
        class_mode='categorical'
    )

    test_generator = test_datagen.flow_from_directory(
        test_dir,  # Path to test data
        target_size=target_size,
        color_mode='grayscale',
        batch_size=32,
        class_mode='categorical'
    )

    # Model definition
    model = build_baseline_model(input_shape)

    # Steps per epoch based on dataset size
    train_steps = train_generator.samples // train_generator.batch_size
    validation_steps = test_generator.samples // test_generator.batch_size

    # Function to reset the generator to avoid running out of data
    def generator_with_reset(generator):
        while True:
            try:
                yield next(generator)
            except StopIteration:
                generator.reset()
                yield next(generator)

    # Wrapping generators with reset functionality
    train_gen_wrapper = generator_with_reset(train_generator)
    test_gen_wrapper = generator_with_reset(test_generator)

    # Measure the training time
    start_time = time.time()

    # Train the model with wrapped generators
    history = model.fit(
        train_gen_wrapper,
        steps_per_epoch=train_steps,
        epochs=10,
        validation_data=test_gen_wrapper,
        validation_steps=validation_steps
    )

    end_time = time.time()
    training_time = end_time - start_time

    return history, training_time

# Train models with different image sizes and record results
results = {}
image_sizes = [(28, 28), (52, 52), (100, 100)]

# Load the pre-trained 28x28 model and its history
results[(28, 28)] = {
    'history': load_and_compile_model('original_neural_network.h5').history,
    'training_time': 51
}

# Train models for other sizes
for size in image_sizes[1:]:
    history, training_time = train_and_evaluate_model(size, (size[0], size[1], 1))
    results[size] = {
        'history': history,
        'training_time': training_time
    }

# Plotting accuracy
epochs = range(1, 11)
colors = {'28x28': 'blue', '52x52': 'green', '100x100': 'red'}

plt.plot(epochs, results[(28, 28)]['history'].history['accuracy'], 'bo-', label='Training accuracy 28x28', color=colors['28x28'])
plt.plot(epochs, results[(28, 28)]['history'].history['val_accuracy'], 'b-', label='Validation accuracy 28x28', color=colors['28x28'])

plt.plot(epochs, results[(52, 52)]['history'].history['accuracy'], 'go-', label='Training accuracy 52x52', color=colors['52x52'])
plt.plot(epochs, results[(52, 52)]['history'].history['val_accuracy'], 'g-', label='Validation accuracy 52x52', color=colors['52x52'])

plt.plot(epochs, results[(100, 100)]['history'].history['accuracy'], 'ro-', label='Training accuracy 100x100', color=colors['100x100'])
plt.plot(epochs, results[(100, 100)]['history'].history['val_accuracy'], 'r-', label='Validation accuracy 100x100', color=colors['100x100'])

plt.title('Training and Validation Accuracy Comparison')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Plotting training times
sizes = ['28x28', '52x52', '100x100']
times = [results[(28, 28)]['training_time'], results[(52, 52)]['training_time'], results[(100, 100)]['training_time']]

plt.bar(sizes, times)
plt.title('Training Time Comparison')
plt.xlabel('Image Size')
plt.ylabel('Time (seconds)')
plt.show()
