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

# Setup data function
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

# Function to build models with varying constraints
def build_model_fewer_constraints():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=(28, 28, 1)),
        tf.keras.layers.Conv2D(40, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(80, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(1000, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def build_model_more_constraints():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=(28, 28, 1)),
        tf.keras.layers.Conv2D(10, (5, 5), activation='tanh', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(25, (5, 5), activation='tanh', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(100, activation='tanh', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Function to train models and plot results
def train_and_plot_models():
    # Setup data
    train_gen, test_gen = setup_data()
    train_steps = train_gen.samples // train_gen.batch_size
    validation_steps = test_gen.samples // test_gen.batch_size

    # Load, compile, and get history of the original model
    original_model = load_and_compile_model('original_neural_network.h5')
    original_history = original_model.history.history

    # Build models with more and fewer constraints
    model_fewer_constraints = build_model_fewer_constraints()
    model_more_constraints = build_model_more_constraints()

    # Wrapping generators with reset functionality
    def generator_with_reset(generator):
        while True:
            try:
                yield next(generator)
            except StopIteration:
                generator.reset()
                yield next(generator)

    train_gen_wrapper = generator_with_reset(train_gen)
    test_gen_wrapper = generator_with_reset(test_gen)

    # Train model with fewer constraints
    history_fewer_constraints = model_fewer_constraints.fit(
        train_gen_wrapper,
        steps_per_epoch=train_steps,
        epochs=10,
        validation_data=test_gen_wrapper,
        validation_steps=validation_steps
    )

    # Train model with more constraints
    history_more_constraints = model_more_constraints.fit(
        train_gen_wrapper,
        steps_per_epoch=train_steps,
        epochs=10,
        validation_data=test_gen_wrapper,
        validation_steps=validation_steps
    )

    # Plotting results
    epochs = range(1, 11)
    plt.figure(figsize=(10, 6))
    
    # Plot original model's performance
    plt.plot(epochs, original_history['accuracy'], 'bo-', label='20-50-500 Training accuracy')
    plt.plot(epochs, original_history['val_accuracy'], 'b-', label='20-50-500 Validation accuracy')
    
    # Plot fewer constraints model's performance
    plt.plot(epochs, history_fewer_constraints.history['accuracy'], 'ro-', label='40-80-1000 Training accuracy')
    plt.plot(epochs, history_fewer_constraints.history['val_accuracy'], 'r-', label='40-80-1000 Validation accuracy')
    
    # Plot more constraints model's performance
    plt.plot(epochs, history_more_constraints.history['accuracy'], 'go-', label='10-25-100 Training accuracy')
    plt.plot(epochs, history_more_constraints.history['val_accuracy'], 'g-', label='10-25-100 Validation accuracy')
    
    plt.title('Training and Validation Accuracy Comparison')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

# Run the function to train and plot models
train_and_plot_models()
