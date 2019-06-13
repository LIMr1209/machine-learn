import torch
import os
from pytorch_pretrained_biggan import (BigGAN, one_hot_from_names, truncated_noise_sample,
                                       save_as_images, display_in_terminal, one_hot_from_int)

# OPTIONAL: if you want to have more information on what's happening, activate the logger as follows
import logging

logging.basicConfig(level=logging.INFO)

# Load pre-trained model tokenizer (vocabulary)
model = BigGAN.from_pretrained('biggan-deep-512')

# Prepare a input
truncation = 0.4
# class_vector = one_hot_from_names(['soap bubble', 'coffee', 'mushroom'], batch_size=3)  # pip install nltk
class_vector = one_hot_from_int([445, 445, 445], batch_size=3)
noise_vector = truncated_noise_sample(truncation=truncation, batch_size=3)

# All in tensors
noise_vector = torch.from_numpy(noise_vector)
class_vector = torch.from_numpy(class_vector)

# If you have a GPU, put everything on cuda
noise_vector = noise_vector.to('cuda')
class_vector = class_vector.to('cuda')
model.to('cuda')

# Generate an image
with torch.no_grad():
    output = model(noise_vector, class_vector, truncation)

# If you have a GPU put back on CPU
output = output.to('cpu')

# If you have a sixtel compatible terminal you can display the images in the terminal
# (see https://github.com/saitoha/libsixel for details)
# display_in_terminal(output)

# Save results as png images
save_as_images(output, file_name='image/')
