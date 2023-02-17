import tensorflow as tf
from keras.layers import Conv2D, BatchNormalization, Dropout, LeakyReLU, MaxPool2D, Input, Concatenate


class PatchGANDiscriminator:
    def __init__(self, input_shape, filters = 64):
        input_shape = input_shape
        initializer = tf.keras.initializers.HeNormal()
        gen_input = Input(shape=input_shape)  #Gen inputs or the satellite images
        gen_output = Input(shape=input_shape)  #Gen outputs or the google maps
        x = Concatenate(axis=-1)([gen_input, gen_output]) #(256, 256, 6)

        x = Conv2D(filters=filters, kernel_size=(4,4), strides = (2,2), padding='same', kernel_initializer=initializer)(x) #Down
        x = LeakyReLU()(x)

        x = Conv2D(filters = filters*2, kernel_size=(4,4), strides = (2,2), padding='same', kernel_initializer=initializer)(x) #Down
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)

        x = Conv2D(filters = filters * 4, kernel_size=(4, 4), strides=(2, 2), padding='same', kernel_initializer=initializer)(x) #Down
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)

        x = Conv2D(filters = filters*8, padding='valid', kernel_size=(4,4), strides = 1, kernel_initializer=initializer)(x) #Stay
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)

        x = Conv2D(filters = 1, padding='valid', kernel_size=(4, 4), strides=1, kernel_initializer=initializer)(x) #prediction layer

        self.discriminator = tf.keras.Model(inputs = [gen_input, gen_output], outputs = x)
        self.discriminator.summary()
        pass



    def get_discriminator(self):
        return self.discriminator