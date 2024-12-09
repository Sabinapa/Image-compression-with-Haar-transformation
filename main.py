from PIL import Image
import numpy as np
import zlib

def load_image(file_path):
    image = Image.open(file_path).convert('L')
    return np.array(image)

def split_into_blocks(image, block_size=8):
    h, w = image.shape
    h_padded = (h + block_size - 1) // block_size * block_size
    w_padded = (w + block_size - 1) // block_size * block_size
    padded_image = np.zeros((h_padded, w_padded), dtype=image.dtype)
    padded_image[:h, :w] = image

    blocks = []
    for i in range(0, h_padded, block_size):
        for j in range(0, w_padded, block_size):
            blocks.append(padded_image[i:i+block_size, j:j+block_size])
    return blocks, (h, w)


image_path = "slike BMP/Lena.bmp"
image = load_image(image_path)
print(f"Dimenzije slike: {image.shape}")

blocks, original_shape = split_into_blocks(image)
print(f"Število blokov: {len(blocks)}")
print(f"Velikost prvega bloka: {blocks[0].shape}")

block = blocks[0]



