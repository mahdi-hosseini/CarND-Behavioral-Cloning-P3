# Imports for keras, sklearn and matplotlib
from keras.models import Sequential, Model
from keras.layers import Cropping2D, Lambda
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import ELU
from keras.callbacks import ModelCheckpoint, Callback
from keras.optimizers import Adam
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import sklearn

# Imports for os, csv, opencv and numpy
import os
import csv
import cv2
import numpy as np

def extract_samples(dir='data/'):
    """Extracts the samples data from CSV file.

    Args:
        dir (str): Directory of the input dataset. (default: data/).

    Returns:
        samples_list (list): An array containing each line of a CSV file.

    Comments:
        The CSV structure should be as follows:
            samples_list[0] = Center Image
            samples_list[1] = Left Image
            samples_list[2] = Right Image
            samples_list[3] = Steering Angle
            samples_list[4] = Throttle
            samples_list[5] = Break
            samples_list[6] = Speed
    """
    samples_list = []
    with open(dir+'driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        next(reader, None)  # skip the headers
        for line in reader:
            samples_list.append(line)
    return samples_list

def get_model(architecture='main'):
    """Generates a Keras model based on input.

    Args:
        architecture (str): Network architecture of choice. (default: main)

    Returns:
        model (keras.model): A keras model if valid architecture was
        chosen. Exception is raised otherwise.

    Raises:
        ValueError: If `architecture` case is invalid.
    """
    if architecture == 'main':
        model = Sequential()

        model.add(Lambda(lambda x: x/127.5 - 1.0,input_shape=(160,320,3)))
        model.add(Cropping2D(cropping=((70,25),(0,0))))

        model.add(Convolution2D(24, 5, 5, border_mode='valid', subsample=(2, 2)))
        model.add(ELU())

        model.add(Convolution2D(36, 5, 5, border_mode='valid', subsample=(2, 2)))
        model.add(ELU())

        model.add(Convolution2D(48, 5, 5, border_mode='valid', subsample=(2, 2)))
        model.add(ELU())

        model.add(Convolution2D(64, 3, 3, border_mode='same', subsample=(1, 1)))
        model.add(ELU())
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))

        model.add(Convolution2D(64, 3, 3, border_mode='same', subsample=(1, 1)))
        model.add(ELU())
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))

        model.add(Flatten())

        model.add(Dense(1164))
        model.add(ELU())
        model.add(BatchNormalization())
        model.add(Dropout(0.5))

        model.add(Dense(100))
        model.add(ELU())
        model.add(BatchNormalization())
        model.add(Dropout(0.5))

        model.add(Dense(50))
        model.add(ELU())
        model.add(BatchNormalization())

        model.add(Dense(10))
        model.add(ELU())
        model.add(BatchNormalization())

        model.add(Dense(1))
    elif architecture == 'udacity':
        model = Sequential()
        model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160,320,3)))
        model.add(Cropping2D(cropping=((70,25),(0,0))))
        model.add(Convolution2D(24,5,5,subsample=(2,2),activation="relu"))
        model.add(Convolution2D(36,5,5,subsample=(2,2),activation="relu"))
        model.add(Convolution2D(48,5,5,subsample=(2,2),activation="relu"))
        model.add(Convolution2D(64,3,3,activation="relu"))
        model.add(Convolution2D(64,3,3,activation="relu"))
        model.add(Flatten())
        model.add(Dense(1164))
        model.add(Dense(100))
        model.add(Dense(50))
        model.add(Dense(10))
        model.add(Dense(1))
    else:
        raise ValueError(architecture+'is an invalid architecture name!')
        return
    return model

def sample_generator(samples, batch_size=32, dir='data/'):
    """Generator to load and preprocess data on the fly,
    in batch size portions to feed to the Behavioral Cloning model.

    Args:
        samples (list): An array containing each line of a CSV file.
        batch_size (int): The number of images that each batch will contain.
        dir (str): Directory of the input dataset. (default: data/).

    Yields:
        X_train, y_train (list, list): A shuffled training sample and
        associated steering angle.

    """
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []

            # A list of correctoion factors
            # 0.0 for center, 0.25 for left, -0.25 for right camera
            angle_correction_factor = [0.0, 0.25, -0.25]

            for batch_sample in batch_samples:
                for i in range(3):
                    name = dir+'IMG/'+batch_sample[i].split('/')[-1]
                    camera_image = cv2.imread(name)
                    angle = float(batch_sample[3])+float(angle_correction_factor[i])
                    images.append(camera_image)
                    angles.append(angle)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

def visualize_loss(model_history_object):
    """Visualizes the performance of deep learning model
    over time during training using Matplotlib.

    Args:
        model_history_object (dict): Model training history object. It contains
        training metrics for each epoch. This includes the loss and the accuracy
        (for classification problems) as well as the loss and accuracy for
        the validation dataset, if one is set.
    """
    ### plot the training and validation loss for each epoch
    plt.plot(model_history_object.history['loss'])
    plt.plot(model_history_object.history['val_loss'])
    plt.title('model mean squared error loss')
    plt.ylabel('mean squared error loss')
    plt.xlabel('epoch')
    plt.legend(['training set', 'validation set'], loc='upper right')
    plt.show()

if __name__ == '__main__':
    samples = []

    # Extract sample names from csv file
    samples = extract_samples()

    # Split dataset into random train and validation subsets
    train_samples, validation_samples = train_test_split(samples, test_size=0.2)

    # Compile and train the model using the generator function
    train_generator = sample_generator(train_samples, batch_size=32)
    validation_generator = sample_generator(validation_samples, batch_size=32)

    # Get the default model
    model = get_model()

    # Compile the model using Adam optimizer with default parameters
    # from in the original paper and Mean Squared Error loss function.
    model.compile(loss='mse', optimizer='adam')

    # Optional: Save the model after every epoch.
    checkpoint = ModelCheckpoint('model{epoch:02d}.h5')

    # Fits the model on data generated batch-by-batch by a `sample_generator()` generator.
    # The samples per epoch is multiplied by 3 to compensate for inclusion left and right cameras.
    history_object = model.fit_generator(train_generator, samples_per_epoch=
                        len(train_samples)*3, validation_data=validation_generator,
                        nb_val_samples=len(validation_samples), nb_epoch=3, verbose=1, callbacks=[checkpoint])

    # Save the Keras model into a single HDF5 file. The Keras model contains:
    #  - the model containing the architecture for future re-creatation
    #  - the weights of the model
    #  - the training configuration (loss, optimizer)
    #  - the state of the optimizer, allowing to resume training where left off
    model.save('model.h5')

    # Plot the mean squared error over training epocs for training and validation sets
    visualize_loss(history_object)

    print('Training successful.')


