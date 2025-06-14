from tensorflow.keras.utils  import image_dataset_from_directory
from tensorflow.keras import layers, models
import tensorflow as tf
from tensorflow.keras.callbacks import CSVLogger
import sys
sys.path.append('../tools')
from callback import SaveBestLossModel, ReduceLearningRate, CustomEarlyStopping, SaveBestAccuModel
from tensorflow.keras.applications import MobileNetV2, InceptionV3, ResNet50V2

def load_data(data_dir='data/tf_data', batch_size=32, image_size=(300, 226)):
    """
    Load the dataset from the specified directory.

    Args:
        data_dir (str): Path to the directory containing the dataset.
        batch_size (int): Number of samples per batch.
        image_size (tuple): Size of the images to be loaded.

    Returns:
        tf.data.Dataset: A dataset object containing the loaded images and labels.
    """
    # Load the datasets
    ds = image_dataset_from_directory(
        data_dir,
        labels='inferred',
        label_mode='int',
        color_mode='grayscale',
        batch_size=batch_size,
        image_size=image_size,
        seed=123,
        interpolation='bilinear',
    )
    return ds


def create_transfer_learning_model(base_model, preprocess_input):
    """
    Create a transfer learning model by adding custom layers on top of the base model.

    Args:
        base_model (tf.keras.Model): The pre-trained base model.
        preprocess_input (function): The preprocessing function for the base model.

    Returns:
        tf.keras.Model: A new model with custom layers added on top of the base model.
    """
    model = models.Sequential([
        # Convert grayscale images to RGB by adding a Conv2D layer with 3 filters
        layers.Conv2D(3, (3, 3), padding='same'),
        # Apply the preprocessing function specific to the base model
        layers.Lambda(preprocess_input),
        # Add the base model
        base_model,
        # Add a global average pooling layer
        layers.GlobalAveragePooling2D(),
        # Add a dense layer with 128 units and ReLU activation
        layers.Dense(128, activation='relu'),
        # Add a dropout layer with a rate of 0.5 to prevent overfitting
        layers.Dropout(0.5),
        # Add the final dense layer with 16 units and softmax activation for classification
        layers.Dense(16, activation='softmax')
    ])

    # Set the name of the model
    model.name = 'freeze_transfert_learn_' + base_model.name

    return model

def create_base_model(base_model, preprocess_input):
    # Add custom layers on top of the base model
    input_shape=base_model.input_shape

    model = models.Sequential([
        layers.Resizing(input_shape[1],input_shape[2], interpolation='bilinear'),
        #layers.Lambda(preprocess_input),
        base_model
        ])
    
    model.name = base_model.name
    return model

def compile_and_fit_model(model, train_ds, val_ds):
    """
    Compile and train the model.

    Args:
        model (tf.keras.Model): The model to be compiled and trained.
        train_ds (tf.data.Dataset): The training dataset.
        val_ds (tf.data.Dataset): The validation dataset.

    Returns:
        tuple: The trained model and the training history.
    """
    # Compile the model with Adam optimizer, sparse categorical crossentropy loss, and accuracy metric
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    # Train the model with the specified datasets and callbacks
    history = model.fit(train_ds, epochs=20,
                        validation_data=val_ds,
                        callbacks=[CSVLogger(model.name + '_log.csv'),
                                   SaveBestAccuModel('best_acc_' + model.name + '.h5'), 
                                   SaveBestLossModel('best_loss_' + model.name + '.h5'),
                                   ReduceLearningRate(),
                                   CustomEarlyStopping()])
    
    # Save the final model
    model.save(f'{model.name}_final.h5')
    
    return model, history

def train_transfer_learning_models(modeles, train_ds, val_ds):
    """
    Train transfer learning models with the given training and validation datasets.

    Args:
        modeles (list): A list of dictionaries containing the base model and preprocessing function.
        train_ds (tf.data.Dataset): The training dataset.
        val_ds (tf.data.Dataset): The validation dataset.
    """
    for model in modeles:
        # Freeze the base model to prevent its weights from being updated during training
        model['base_model'].trainable = False
        # Create a transfer learning model by adding custom layers on top of the base model
        transfert_model = create_transfer_learning_model(model['base_model'], model['preprocess_input'])
        # Compile and train the transfer learning model
        compile_and_fit_model(transfert_model, train_ds, val_ds)

        print(f'{model["base_model"].name} model trained successfully')

def train_base_models(modeles, train_ds, val_ds):
    """
    Train base models with the given training and validation datasets.

    Args:
        modeles (list): A list of dictionaries containing the base model and preprocessing function.
        train_ds (tf.data.Dataset): The training dataset.
        val_ds (tf.data.Dataset): The validation dataset.
    """
    for model in modeles:
        # Allow the base model's weights to be updated during training
        model['base_model'].trainable = True
        # Create a base model with custom layers
        base_model = create_base_model(model['base_model'], model['preprocess_input'])
        # Compile and train the base model
        compile_and_fit_model(base_model, train_ds, val_ds)
        print(f'{model["base_model"].name} model trained successfully')

def create_custom_model():
    """
    Create a custom CNN model for image classification.

    Returns:
        tf.keras.Model: A custom CNN model.
    """
    model = models.Sequential([
        # Resize the input images to the specified size
        layers.Resizing(500, 380),
        # Rescale the pixel values to the range [0, 1]
        layers.Rescaling(1./255),
        # First convolutional layer with 32 filters and ReLU activation
        layers.Conv2D(32, (3, 3), activation='relu'),
        # Max pooling layer to reduce the spatial dimensions
        layers.MaxPooling2D((2, 2)),
        # Second convolutional layer with 64 filters and ReLU activation
        layers.Conv2D(64, (3, 3), activation='relu'),
        # Max pooling layer to reduce the spatial dimensions
        layers.MaxPooling2D((2, 2)),
        # Third convolutional layer with 64 filters and ReLU activation
        layers.Conv2D(64, (3, 3), activation='relu'),
        # Flatten the output of the convolutional layers to feed into the dense layers
        layers.Flatten(),
        # Dense layer with 64 units and ReLU activation
        layers.Dense(64, activation='relu'),
        # Final dense layer with 16 units and softmax activation for classification
        layers.Dense(16, activation='softmax')
    ])
    # Set the name of the model
    model.name = 'custom_model'
    return model

train_ds=load_data('data/tf_data')
val_ds=load_data('data/tf_data_val')

transfer_models = [
    {
        'base_model': tf.keras.applications.MobileNetV2(include_top=False),
        'preprocess_input': tf.keras.applications.mobilenet.preprocess_input
    }
    {
        'base_model': tf.keras.applications.InceptionV3(cinclude_top=False),
        'preprocess_input': tf.keras.applications.inception_v3.preprocess_input
    },
    {
        'base_model': tf.keras.applications.ResNet50V2(include_top=False),
        'preprocess_input': tf.keras.applications.resnet.preprocess_input
    }
 ]

base_models = [
    {
        'base_model': tf.keras.applications.MobileNetV2(classes=16,weights=None,input_shape=(300,266,1),),
        'preprocess_input': tf.keras.applications.mobilenet.preprocess_input
    },
    {
        'base_model': tf.keras.applications.InceptionV3(classes=16,weights=None,input_shape=(300,266,1)),
        'preprocess_input': tf.keras.applications.inception_v3.preprocess_input
    },
    {
        'base_model': tf.keras.applications.ResNet50V2(classes=16,weights=None,input_shape=(300,266,1)),
        'preprocess_input': tf.keras.applications.resnet.preprocess_input
    }
]

train_transfer_learning_models(transfer_models, train_ds, val_ds)

train_base_models(base_models, train_ds, val_ds)