import Dataset_builder
import UNetGenerator
import PatchGANDiscriminator
import ImageTranslationGAN

dataset_directory_path = "C:/Users/allan/Downloads/GANFacesDateset"
target_size = (256,256)
batch_size = 8
epochs = 10
gen_filters = 32
disc_filters = 64
reconstruction_loss_weight = 100

dataset_builder = Dataset_builder.Dataset_builder(directory_path=dataset_directory_path, batch_size=batch_size, target_size=target_size)
dataset = dataset_builder.get_dataset()

# batch_edges = None
#
# for gen_target_outputs in dataset.take(5):
#     batch_edges = np.zeros_like(gen_target_outputs)
#     batch_edges = batch_edges[:,:,:,0]
#     batch_size = gen_target_outputs.shape[0]
#     for i in range(batch_size):
#         single_image = gen_target_outputs[i]
#         image_tensor = single_image.numpy()
#         blurred_image = cv2.GaussianBlur(image_tensor, (3, 3), 0)
#         edge_image = cv2.Canny(blurred_image, threshold1=140, threshold2=140)
#         batch_edges[i] = edge_image
#         pass
#     pass
#
# batch_edges = tf.expand_dims(batch_edges, axis = -1)
# batch_edges = tf.tile(batch_edges, [1,1,1,3])



discriminator_builder = PatchGANDiscriminator.PatchGANDiscriminator(filters=disc_filters)
discriminator = discriminator_builder.get_discriminator()

#Build and get the generator
generator_builder = UNetGenerator.Generator()
#generator = generator_builder.get_generator(output_channels=3)
generator = generator_builder.build_Unet_generator(filters=gen_filters)


#Set up the GAN
GAN_builder = ImageTranslationGAN.GAN(reconstruction_loss_weight=reconstruction_loss_weight, generator=generator, discriminator=discriminator)
#Initialize optimizers and loss functions
GAN_builder.initialize_optimizers()
GAN_builder.initialize_loss_function()
GAN_builder.train_model(epochs=epochs, dataset=dataset)



