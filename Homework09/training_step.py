"""
The training_step function

Created: 15.01.22, 19:16

Author: LDankert
"""

import tensorflow as tf
import datetime


def training_step(train_data, generator, discriminator, batch_size, latent_space, loss_function, optimizer):
    noise = tf.random.normal(shape=(batch_size, latent_space))

    for image in train_data:
        with tf.GradientTape() as generator_tape, tf.GradientTape() as discriminator_tape:

            gen_images = generator(noise, training=True)

            output_real = discriminator(image[0], training=True)
            output_fake = discriminator(gen_images, training=True)

            loss_real = loss_function(tf.ones_like(output_real), output_real)
            loss_fake = loss_function(tf.zeros_like(output_fake), output_fake)
            loss_discriminator = loss_real + loss_fake

            loss_generator = loss_function(output_fake, tf.ones_like(output_fake))

        generator_gradient = generator_tape.gradient(loss_generator, generator.trainable_variables)
        discriminator_gradient = discriminator_tape.gradient(loss_discriminator, discriminator.trainable_variables)

        optimizer.apply_gradients(zip(generator_gradient, generator.trainable_variables))
        optimizer.apply_gradients(zip(discriminator_gradient, discriminator.trainable_variables))

    return loss_discriminator, loss_generator, gen_images



