import os
import sys
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
        if sum_idx % 2 == 1:
            for i in range(max(0, sum_idx - cols + 1), min(sum_idx + 1, rows)):
                result.append(input_matrix[i, sum_idx - i])
        else:
            for i in range(min(sum_idx, rows - 1), max(-1, sum_idx - cols, -1), -1):
                result.append(input_matrix[i, sum_idx - i])
    return np.array(result)

def apply_threshold(data, threshold):
    return np.where(data < threshold, 0, data)

def entropy_encode_with_library(data):
    data_bytes = ','.join(map(str, data)).encode('utf-8')
    encoded_data = zlib.compress(data_bytes)
    return encoded_data

#############################################################################

def inverse_zigzag(input_array, rows, cols):
    result = np.zeros((rows, cols))
    idx = 0
    for sum_idx in range(rows + cols - 1):
        if sum_idx % 2 == 1:
            for i in range(max(0, sum_idx - cols + 1), min(sum_idx + 1, rows)):
                result[i, sum_idx - i] = input_array[idx]
                idx += 1
        else:
            for i in range(min(sum_idx, rows - 1), max(-1, sum_idx - cols, -1), -1):
                result[i, sum_idx - i] = input_array[idx]
                idx += 1
    return result

def inverse_haar_transform(block):
    H = haar_matrix()
    return H @ block @ H.T

def assemble_image_from_blocks(blocks, image_shape, block_size=8):
    h, w = image_shape
    image = np.zeros((h, w), dtype=np.uint8)
    block_idx = 0
    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            block = np.clip(np.rint(blocks[block_idx]), 0, 255).astype(np.uint8)
            image[i:i+block_size, j:j+block_size] = block
            block_idx += 1
    return image

def decompress(input_file, output_file, block_size=8):
    with open(input_file, "rb") as file:
        dimensions = file.readline().decode('utf-8').strip()
        #print(f"Prebrane dimenzije: {dimensions}")
        original_shape = tuple(map(int, dimensions.split(',')))
        compressed_data = file.read()

    #print(f"Velikost prebranih podatkov: {len(compressed_data)} bajtov")
    #print(blocks_data)

    blocks = []
    offset = 0
    expected_blocks = (original_shape[0] // block_size) * (original_shape[1] // block_size)
    #print("Koliko blokov bi naj bilo", expected_blocks)

    for idx in range(expected_blocks):
        # Dekodiramo dolžino bloka z `zlib.decompress`
        decoded_data = zlib.decompress(compressed_data[offset:])
        decoded_str = decoded_data.decode('utf-8')
        block_1d = list(map(float, decoded_str.split(',')))

        # Inverzna cik-cak pretvorba
        block_2d = inverse_zigzag(block_1d, block_size, block_size)

        # Inverzna Haarova transformacija
        restored_block = inverse_haar_transform(block_2d)
        blocks.append(restored_block)

        offset += len(zlib.compress(decoded_str.encode('utf-8')))

        '''
        if idx < 5:
            print(f"Blok {idx + 1}:")
            print("1D polje po dekompresiji:")
            print(block_1d)
            print("2D polje po inverzni cik-cak pretvorbi:")
            print(block_2d)
            print("2D polje po inverzni Haarovi transformaciji:")
            print(restored_block)
            print("-----------")
        '''

    restored_image = assemble_image_from_blocks(blocks, original_shape, block_size)
    Image.fromarray(restored_image).save(output_file)
    print(f"Restavrirana slika je shranjena v '{output_file}'")

def compress(image_path, output_file, threshold):
    image = load_image(image_path)

    blocks, original_shape = split_into_blocks(image)
    #print(f"Velikost prvega bloka: {blocks[0].shape}")

    with open(output_file, "wb") as file:
        file.write(f"{image.shape[0]},{image.shape[1]}\n".encode('utf-8'))

        for idx, block in enumerate(blocks):
            # Haarova transformacija
            transformed_block = haar_transform(block)

            # Cik-cak pretvorba
            zigzagged = zigzag(transformed_block)

            #print("Tip podatkov v `zigzagged`:", type(zigzagged), "Prva vrednost:", zigzagged[0])
            #print("Tip podatkov v `threshold`:", type(threshold), "Vrednost:", threshold)

            # Prag stiskanja
            thresholded = apply_threshold(zigzagged, threshold)

            # Entropijsko kodiranje
            encoded_data = entropy_encode_with_library(thresholded)

            # Zapiši podatke v datoteko
            file.write(encoded_data)

            '''
            if idx < 5:  # Prikaz podatkov za prvih 5 blokov
                print(f"Blok {idx + 1}:")
                print("Originalni blok (8x8):")
                print(block)
                print("Transformirani blok (Haarova transformacija):")
                print(transformed_block)
                print("Cik-cak pretvorba (1D polje):")
                print(zigzagged)
                print("Po pragu stiskanja:")
                print(thresholded)
                print(f"Velikost kodiranih podatkov: {len(encoded_data)} bajtov")
                print("-----------")
            '''

    print(f"Vsi kodirani podatki so shranjeni v datoteko '{output_file}'")


##############################################################

def main():
    if len(sys.argv) < 4:
        print("Napaka: Premalo argumentov.")
        print("Uporaba: dn2 <vhodna_datoteka> <c|d> <izhodna_datoteka> [<prag>]")
        sys.exit(1)

    input_file = sys.argv[2]
    option = sys.argv[3]
    output_file = sys.argv[4]
    if option == "c":
        threshold = float(sys.argv[5]) if len(sys.argv) > 4 else 0.0


    print(input_file, option, output_file)

    if not os.path.exists(input_file):
        print(f"Napaka: Vhodna datoteka '{input_file}' ne obstaja.")
        sys.exit(1)

    if option == "c":
        if not input_file.lower().endswith(".bmp"):
            print("Napaka: Za kompresijo je podprta le vhodna datoteka BMP.")
            sys.exit(1)
        compress(input_file, output_file, threshold)
    elif option == "d":
        decompress(input_file, output_file)
    else:
        print("Napaka: Neznana opcija. Uporabite 'c' za kompresijo ali 'd' za dekompresijo.")
        sys.exit(1)


if __name__ == "__main__":
    main()




