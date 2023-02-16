import tensorflow as tf
import matplotlib.pyplot as plt
import random
import time
import numpy as np
import cv2

class GAN:
    def __init__(self, reconstruction_loss_weight, generator, discriminator):
        self.reconstruction_loss_weight = reconstruction_loss_weight
        self.generator = generator
        self.discriminator = discriminator
        pass

    def generate_images(self, model, test_input, tar):
        index = random.randrange(test_input.shape[0])

        prediction = model(test_input, training=True)
        plt.figure(figsize=(15, 15))

        display_list = [test_input[index], tar[index], prediction[index]]
        title = ['Generator input', 'Generator output y_true', 'Generator output y_pred']

        for i in range(3):
            plt.subplot(1, 3, i + 1)
            plt.title(title[i])
            # Getting the pixel values in the [0, 1] range to plot.
            plt.imshow(display_list[i] * 0.5 + 0.5)
            plt.axis('off')
            pass
        plt.show()
        pass

    def initialize_loss_function(self):
        self.adversarial_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        self.reconstruction_loss = tf.keras.losses.MeanAbsoluteError()
        pass

    def initialize_optimizers(self):
        self.gen_optimizer = tf.keras.optimizers.Adam(0.0002, beta_1=0.5)
        self.disc_optimizer = tf.keras.optimizers.Adam(0.0002, beta_1=0.5)
        pass

    def get_gen_loss(self, fake_disc_outputs, fake_gen_outputs, gen_target_outputs):
        # The target labels for these should be 1s
        fake_disc_output_labels = tf.ones_like(fake_disc_outputs)
        # Calculate adversarial loss
        adversarial_loss = self.adversarial_loss(y_true=fake_disc_output_labels, y_pred=fake_disc_outputs)
        #Calculate the reconstruction loss
        reconstruction_loss = self.reconstruction_loss(y_true = gen_target_outputs, y_pred = fake_gen_outputs)
        #Calculate the total loss
        total_gen_loss = adversarial_loss + (self.reconstruction_loss_weight * reconstruction_loss)
        return total_gen_loss

    def get_disc_loss(self, real_disc_outputs, fake_disc_outputs):
        # Create labels
        real_disc_output_labels = tf.ones_like(real_disc_outputs)
        # Calculate the real loss
        real_loss = self.adversarial_loss(real_disc_output_labels, real_disc_outputs)

        # Get target labels for these outputs
        fake_disc_output_targets = tf.zeros_like(fake_disc_outputs)
        # Calculate the fake loss
        generated_loss = self.adversarial_loss(fake_disc_output_targets, fake_disc_outputs)

        #Calculate the total loss
        total_disc_loss = real_loss + generated_loss
        return total_disc_loss

    def preprocess_tensor(self, input_tensor):
        processed_tensor = tf.convert_to_tensor(input_tensor)
        processed_tensor = tf.cast(processed_tensor, tf.float32)
        processed_tensor = (processed_tensor - 127.5) / 127.5
        return processed_tensor


    def get_real_gen_inputs(self, gen_target_outputs):
        real_gen_inputs = np.zeros_like(gen_target_outputs)
        real_gen_inputs = real_gen_inputs[:, :, :, 0]
        batch_size = gen_target_outputs.shape[0]
        for i in range(batch_size):
            single_image = gen_target_outputs[i]
            image_tensor = single_image.numpy()
            blurred_image = cv2.GaussianBlur(image_tensor, (3, 3), 0)
            edge_image = cv2.Canny(blurred_image, threshold1=140, threshold2=140)
            real_gen_inputs[i] = edge_image
            pass

        real_gen_inputs = tf.expand_dims(real_gen_inputs, axis=-1)
        real_gen_inputs = tf.tile(real_gen_inputs, [1, 1, 1, 3])
        real_gen_inputs = self.preprocess_tensor(real_gen_inputs)
        return real_gen_inputs

    def train_model(self, epochs, dataset):
        for epoch in range(epochs):
            for step, gen_target_outputs in enumerate(dataset):
                num_batches = len(dataset)
                start_time = time.time()
                with tf.GradientTape() as discriminator_tape, tf.GradientTape() as generator_tape:
                    #Get the real gen inputs
                    real_gen_inputs = self.get_real_gen_inputs(gen_target_outputs=gen_target_outputs)
                    #Preprocess the target outputs
                    gen_target_outputs = self.preprocess_tensor(gen_target_outputs)

                    fake_gen_outputs = self.generator(real_gen_inputs, training=True)
                    # Pass the fake gen outputs with real gen inputs through the discriminator
                    fake_disc_outputs = self.discriminator([real_gen_inputs, fake_gen_outputs], training=True)
                    # Using the gen_target outputs, get the real disc outputs
                    real_disc_outputs = self.discriminator([real_gen_inputs, gen_target_outputs], training=True)
                    # Using the real gen inputs and the target outputs, calculate the discriminator loss
                    total_disc_loss = self.get_disc_loss(fake_disc_outputs=fake_disc_outputs, real_disc_outputs=real_disc_outputs)

                    # Using the gen target outputs and real gen inputs, calculate the generator loss
                    total_gen_loss = self.get_gen_loss(gen_target_outputs=gen_target_outputs,fake_gen_outputs=fake_gen_outputs,fake_disc_outputs=fake_disc_outputs)
                    pass

                    # Differentiate the total disc loss with respect to the disc weights
                    disc_gradient = discriminator_tape.gradient(total_disc_loss, self.discriminator.trainable_variables)
                    # Gradient descent using the gradient
                    self.disc_optimizer.apply_gradients(zip(disc_gradient, self.discriminator.trainable_variables))

                    # Differentiate the gradient with respect to the generator trainable weights
                    gen_gradient = generator_tape.gradient(total_gen_loss, self.generator.trainable_variables)
                    # Gradient descent using the gradient
                    self.gen_optimizer.apply_gradients(zip(gen_gradient, self.generator.trainable_variables))
                    end_time = time.time()
                    time_per_step = end_time - start_time
                    est_epoch_time = time_per_step * num_batches
                    print("Step:{4}/{5} Total gen loss = {0:.4f} || Total disc loss = {1:.4f} || Time Per step = {2:.4f}s || Est. Time per epoch = {3:.4f}s"
                          .format(total_gen_loss, tf.math.reduce_mean(total_disc_loss), time_per_step, est_epoch_time, step, num_batches))
                    if step % 50 == 0:
                        self.generate_images(self.generator, real_gen_inputs, gen_target_outputs)
                    pass
                pass
            pass
        pass



    pass

