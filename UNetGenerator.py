import tensorflow as tf
from keras.layers import Input, Conv2D, Conv2DTranspose, LeakyReLU, BatchNormalization, MaxPool2D, Concatenate


class Generator:
    def __init__(self):
        pass

    def encoding_block(self, input, first_block, filters, kernel_size):
        #Conv once
        x = Conv2D(kernel_size=kernel_size, filters = filters, strides = 1, padding = 'same', kernel_initializer='he_normal', use_bias=False)(input)
        if not first_block:
            x = BatchNormalization()(x)
            pass
        x = LeakyReLU()(x)
        # #Conv second time
        # x = Conv2D(kernel_size=kernel_size, filters=filters, strides = 1, padding='same', kernel_initializer='he_normal', use_bias=False)(x)
        # if not first_block:
        #     x = BatchNormalization()(x)
        #     pass
        # x = LeakyReLU()(x)
        #save the second conv
        skip_connection = x
        #downsample it
        down_sampled_layer = MaxPool2D(pool_size=(2,2), strides=(2,2))(x)

        return down_sampled_layer, skip_connection

    def decoding_block(self, input_layer, skip_connection, skip_con_filters, kernel_size):
        # Conv once
        x = Conv2D(kernel_size=kernel_size, filters=skip_con_filters*2, strides = 1, padding='same', kernel_initializer='he_normal', use_bias=False)(input_layer)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)
        # # Conv twice
        # x = Conv2D(kernel_size=kernel_size, filters=skip_con_filters*2, strides=1, padding='same', kernel_initializer='he_normal', use_bias=False)(x)
        # x = BatchNormalization()(x)
        # x = LeakyReLU()(x)
        #Upsample the input and half filters
        x = Conv2DTranspose(filters=skip_con_filters, kernel_size=kernel_size, strides=2, padding='same', kernel_initializer='he_normal', use_bias=False)(x)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)
        #Concat with the skip connection
        concat_layer = Concatenate(axis=-1)([skip_connection, x])
        return concat_layer

    def final_decoding_block(self, input_layer, kernel_size, filters, output_channels):
        x = Conv2D(kernel_size=kernel_size, filters=filters, strides=1, padding='same',kernel_initializer='he_normal', use_bias=False)(input_layer)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)

        x = Conv2D(kernel_size=kernel_size, filters=filters, strides=1, padding='same', kernel_initializer='he_normal',use_bias=False)(x)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)

        x = Conv2D(kernel_size=1, filters=output_channels, strides=1, padding='same', kernel_initializer='he_normal',use_bias=False, activation='tanh')(x)
        return x

    def build_Unet_generator(self, filters, input_shape):
        # Encoder blocks
        input_layer = Input(shape=input_shape)
        block1, skip1 = self.encoding_block(input=input_layer, first_block=True, kernel_size=(4, 4),filters=filters)
        block2, skip2 = self.encoding_block(input=block1, first_block=False, kernel_size=(4, 4),filters=filters*2)
        block3, skip3 = self.encoding_block(input=block2, first_block=False, kernel_size=(4, 4),filters=filters*4)
        block4, skip4 = self.encoding_block(input=block3, first_block=False, kernel_size=(4, 4),filters=filters*8)
        block5, skip5 = self.encoding_block(input=block4, first_block=False, kernel_size=(4, 4),filters=filters*8)
        block6, skip6 = self.encoding_block(input=block5, first_block=False, kernel_size=(4, 4),filters=filters*8)
        block7, skip7 = self.encoding_block(input=block6, first_block=False, kernel_size=(4, 4),filters=filters*8)

        # Decoder blocks
        block8 = self.decoding_block(input_layer=block7, skip_connection=skip7, skip_con_filters=filters*8,kernel_size=(4, 4))
        block9 = self.decoding_block(input_layer=block8, skip_connection=skip6, skip_con_filters=filters*8,kernel_size=(4, 4))
        block10 = self.decoding_block(input_layer=block9, skip_connection=skip5, skip_con_filters=filters*8,kernel_size=(4, 4))
        block11 = self.decoding_block(input_layer=block10, skip_connection=skip4, skip_con_filters=filters*8,kernel_size=(4, 4))
        block12 = self.decoding_block(input_layer=block11, skip_connection=skip3, skip_con_filters=filters*4,kernel_size=(4, 4))
        block13 = self.decoding_block(input_layer=block12, skip_connection=skip2, skip_con_filters=filters*2,kernel_size=(4, 4))
        block14 = self.decoding_block(input_layer=block13, skip_connection=skip1, skip_con_filters=filters,kernel_size=(4, 4))

        final_layer = self.final_decoding_block(input_layer=block14, kernel_size=(4, 4), filters=32,output_channels=3)
        generator_model = tf.keras.Model(inputs=input_layer, outputs = final_layer)
        generator_model.summary()
        return generator_model