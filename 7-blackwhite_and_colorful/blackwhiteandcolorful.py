import os
import shutil
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import json

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

# Function to invert image colors
def invert_colors(image):
    return 1.0 - image

# Function to train and evaluate models with different color schemes
def train_and_evaluate_model(train_dir, test_dir, input_shape, augment_data=True, invert=False):
    # Data augmentation and normalization for training
    train_datagen = ImageDataGenerator(
        rescale=1./255,  # Rescale images from [0,255] to [0,1]
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        preprocessing_function=invert_colors if invert else None
    )
    test_datagen = ImageDataGenerator(rescale=1./255, preprocessing_function=invert_colors if invert else None)  # Only rescale for test data

    # Load data from combined directories
    train_generator = train_datagen.flow_from_directory(
        train_dir,  # Path to training data
        target_size=(28, 28),
        color_mode='grayscale',
        batch_size=32,
        class_mode='categorical'
    )

    test_generator = test_datagen.flow_from_directory(
        test_dir,  # Path to test data
        target_size=(28, 28),
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

    # Train the model with wrapped generators
    history = model.fit(
        train_gen_wrapper,
        steps_per_epoch=train_steps,
        epochs=10,
        validation_data=test_gen_wrapper,
        validation_steps=validation_steps
    )

    return history

# Consolidate images
train_dir, test_dir = consolidate_images()

# Train models with different color schemes and record results
results = {}

# Load the pre-trained baseline model and its history
results['baseline'] = {
    'history': load_and_compile_model('original_neural_network.h5').history
}

# Train model with inverted colors
history_inverted = train_and_evaluate_model(train_dir, test_dir, (28, 28, 1), invert=True)
results['inverted'] = {
    'history': history_inverted
}

# Plotting accuracy
epochs = range(1, 11)

# Plot baseline model accuracy
plt.plot(epochs, results['baseline']['history'].history['accuracy'], 'bo-', label='Baseline Training accuracy')
plt.plot(epochs, results['baseline']['history'].history['val_accuracy'], 'b-', label='Baseline Validation accuracy')

# Plot inverted colors model accuracy
plt.plot(epochs, results['inverted']['history'].history['accuracy'], 'ro-', label='Inverted Training accuracy')
plt.plot(epochs, results['inverted']['history'].history['val_accuracy'], 'r-', label='Inverted Validation accuracy')

plt.title('Training and Validation Accuracy Comparison')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
