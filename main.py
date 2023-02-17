from tensorflow.python.feature_column.feature_column import input_layer

import Dataset_builder
import UNetGenerator
import PatchGANDiscriminator
import ImageTranslationGAN
import tensorflow as tf

# from keras import mixed_precision
# policy = mixed_precision.Policy('mixed_float16')
# mixed_precision.set_global_policy(policy)

dataset_directory_path = "C:/Users/allan/Downloads/GANFacesDateset"
target_size = (256,256)
input_shape = (256,256,3)
batch_size =12
epochs = 20
gen_filters = 32
disc_filters = 64
reconstruction_loss_weight = 100

dataset_builder = Dataset_builder.Dataset_builder(directory_path=dataset_directory_path, batch_size=batch_size, target_size=target_size)
dataset = dataset_builder.get_dataset()



discriminator_builder = PatchGANDiscriminator.PatchGANDiscriminator(filters=disc_filters, input_shape=input_shape)
discriminator = discriminator_builder.get_discriminator()

#Build and get the generator
generator_builder = UNetGenerator.Generator()
#generator = generator_builder.get_generator(output_channels=3)
generator = generator_builder.build_Unet_generator(filters=gen_filters, input_shape=input_shape)


# generator = tf.keras.models.load_model("Models/Epoch5/Generator")
# discriminator = tf.keras.models.load_model("Models/Epoch5/Discriminator")

#Set up the GAN
GAN_builder = ImageTranslationGAN.GAN(reconstruction_loss_weight=reconstruction_loss_weight, generator=generator, discriminator=discriminator)
#Initialize optimizers and loss functions
GAN_builder.initialize_optimizers()
GAN_builder.initialize_loss_function()
GAN_builder.train_model(epochs=epochs, dataset=dataset)



