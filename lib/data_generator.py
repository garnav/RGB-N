import data_manipulation
import numpy as np
import skimage.transform
import keras

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, batch_size, dim=(100,7,7,1024), invert=False, shuffle=False):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.list_IDs = list_IDs
        self.shuffle = shuffle
        self.invert = invert
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples'
        # Initialization
        X = np.empty((self.batch_size, *self.dim))
        y = np.empty((self.batch_size, self.dim[0]), dtype=np.int)

        for i, ID in enumerate(list_IDs_temp):
            X[i, ] = data_manipulation.load_x(ID)

            loaded_y = data_manipulation.load_y(ID)
            if invert:
                all_ones = loaded_y == 1
                all_zeros = loaded_y == 0
                loaded_y[all_ones] = 0
                loaded_y[all_zeros] = 1

            y[i, ] = loaded_y

        return X, y
