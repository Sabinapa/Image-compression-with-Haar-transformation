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

def haar_matrix():
    H = np.array([
        [np.sqrt(8/64), np.sqrt(8/64), 1/2, 0, np.sqrt(2/4), 0, 0, 0],
        [np.sqrt(8/64), np.sqrt(8/64), 1/2, 0, -np.sqrt(2/4), 0, 0, 0],
        [np.sqrt(8/64), np.sqrt(8/64), -1/2, 0, 0, np.sqrt(2/4), 0, 0],
        [np.sqrt(8/64), np.sqrt(8/64), -1/2, 0, 0, -np.sqrt(2/4), 0, 0],
        [np.sqrt(8/64), -np.sqrt(8/64), 0, 1/2, 0, 0, np.sqrt(2/4), 0],
        [np.sqrt(8/64), -np.sqrt(8/64), 0, 1/2, 0, 0, -np.sqrt(2/4), 0],
        [np.sqrt(8/64), -np.sqrt(8/64), 0, -1/2, 0, 0, 0, np.sqrt(2/4)],
        [np.sqrt(8/64), -np.sqrt(8/64), 0, -1/2, 0, 0, 0, -np.sqrt(2/4)],
    ])
    return H

def haar_transform(block):
    H = haar_matrix()
    return H.T @ block @ H

def zigzag(input_matrix):
    rows, cols = input_matrix.shape
    result = []
    for sum_idx in range(rows + cols - 1):
        if sum_idx % 2 == 1:  # Gremo od zgoraj levo proti spodaj desno
            for i in range(max(0, sum_idx - cols + 1), min(sum_idx + 1, rows)):
                result.append(input_matrix[i, sum_idx - i])
        else:  # Gremo od spodaj desno proti zgoraj levo
            for i in range(min(sum_idx, rows - 1), max(-1, sum_idx - cols, -1), -1):
                result.append(input_matrix[i, sum_idx - i])
    return np.array(result)

image_path = "slike BMP/Lena.bmp"
image = load_image(image_path)
print(f"Dimenzije slike: {image.shape}")

blocks, original_shape = split_into_blocks(image)
print(f"Število blokov: {len(blocks)}")
print(f"Velikost prvega bloka: {blocks[0].shape}")

block = blocks[0]

print(block)
# Izvedemo Haarovo transformacijo
transformed_block = haar_transform(block)
print("Haarova transformacija za prvi blok:")
print(transformed_block)

zigzagged = zigzag(transformed_block)
print("Cik-cak pretvorba (1D polje):")
print(zigzagged)

