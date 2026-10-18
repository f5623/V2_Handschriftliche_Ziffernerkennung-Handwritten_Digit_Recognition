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

# Function to build a model with a specific activation function
def build_model_with_activation(activation):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Input(shape=(28, 28, 1)))
    model.add(tf.keras.layers.Conv2D(20, (5, 5), activation=activation))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.Conv2D(50, (5, 5), activation=activation))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(500, activation=activation))
    model.add(tf.keras.layers.Dense(10, activation='softmax'))
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Setup data function (same as your baseline script)
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

# Function to train models and plot results
def train_and_plot_models():
    # Setup data
    train_gen, test_gen = setup_data()

    # Steps per epoch based on dataset size
    train_steps = train_gen.samples // train_gen.batch_size
    validation_steps = test_gen.samples // test_gen.batch_size

    # Load, compile, and possibly train the original model
    original_model = load_and_compile_model('original_neural_network.h5')

    # Get the original model's history
    original_history = original_model.history.history
    
    # Build models with different activation functions
    model_relu = build_model_with_activation('relu')
    model_sigmoid = build_model_with_activation('sigmoid')

    # Define reset function to avoid running out of data
    def generator_with_reset(generator):
        while True:
            try:
                yield next(generator)
            except StopIteration:
                generator.reset()
                yield next(generator)

    # Wrapping generators with reset functionality
    train_gen_wrapper = generator_with_reset(train_gen)
    test_gen_wrapper = generator_with_reset(test_gen)

    # Train relu model and store history
    history_relu = model_relu.fit(
        train_gen_wrapper,
        steps_per_epoch=train_steps,
        epochs=10,
        validation_data=test_gen_wrapper,
        validation_steps=validation_steps
    )

    # Train sigmoid model and store history
    history_sigmoid = model_sigmoid.fit(
        train_gen_wrapper,
        steps_per_epoch=train_steps,
        epochs=10,
        validation_data=test_gen_wrapper,
        validation_steps=validation_steps
    )

    # Plotting accuracy
    epochs = range(1, 11)
    
    plt.plot(epochs, original_history['accuracy'], 'bo-', label='tanh Training accuracy')
    plt.plot(epochs, original_history['val_accuracy'], 'b-', label='tanh Validation accuracy')

    plt.plot(epochs, history_relu.history['accuracy'], 'ro-', label='relu Training accuracy')
    plt.plot(epochs, history_relu.history['val_accuracy'], 'r-', label='relu Validation accuracy')

    plt.plot(epochs, history_sigmoid.history['accuracy'], 'go-', label='sigmoid Training accuracy')
    plt.plot(epochs, history_sigmoid.history['val_accuracy'], 'g-', label='sigmoid Validation accuracy')

    plt.title('Training and Validation Accuracy Comparison')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

# Run the function to train and plot models
train_and_plot_models()
