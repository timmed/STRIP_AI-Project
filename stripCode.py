import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import ipywidgets
#plt.ion()

from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from IPython.display import display
import matplotlib_inline.backend_inline


import os
cwd = os.getcwd()
print("current working directory", cwd)
import pandas as pd

import cv2
from tqdm.notebook import tqdm
from PIL import Image
OPENSLIDE_PATH = r'C:\Users\Danie\openslide-win64-20220811\\bin'
if hasattr(os, 'add_dll_directory'):
    # Python >= 3.8 on Windows
    with os.add_dll_directory(OPENSLIDE_PATH):
        import openslide
else:
    import openslide

#Image.MAX_IMAGE_PIXELS = None
#print('libraries imported')


train_csv = pd.read_csv('train.csv')
#display(train_csv)
train_csv['image_path'] = train_csv.image_id.apply(lambda x: os.path.join('train', x+".tif"))
columns = train_csv.columns
#print(train_csv)

#print(columns)

def enhance_df(df):
    df["image_size"] = df.image_path.apply(lambda x: Image.open(x).size)
    df["image_pixels"] = df["image_size"].apply(lambda x: int(x[0] * int(x[1])))
    df["image_width"] = df["image_size"].apply(lambda x: int(x[0]))
    df["image_height"] = df["image_size"].apply(lambda x: int(x[1]))
    df["aspect_ratio"] = df["image_width"] / df["image_height"]
    return df


train_csv = enhance_df(train_csv)
print('enhanced train dataframe ready')
#print(train_csv)

#display(train_csv[train_csv.image_pixels==train_csv.image_pixels.max()])
#display(train_csv[train_csv.image_pixels==train_csv.image_pixels.min()])


def tile_resize_stich(row, horizontal_size=4000, cutoff_size=3500000000, tiles_per_side=5, show=False, debug=False):
    '''
    Break up the large image, resize individual tiles, put them back together
    Keep horizontal size at the same size specified for general resizing.
    '''

    # def demarc():
    #     print('=' * 100)

    # get the metadata -----------------------------------------------------------
    image_path = row.image_path
    image_width = row.image_width
    image_height = row.image_height

    # Show the original image, if req'd ------------------------------------------
    # if show and row.image_pixels<cutoff_size:
    #     orig_image = np.array(cv2.imread(row.image_path))
    #     orig_image = cv2.cvtColor(orig_image, cv2.COLOR_RGB2BGR)
    #     print('Original Image')
    #     plt.imshow(orig_image); plt.show()
    #     demarc()

    # What should be the tile size? ----------------------------------------------
    # Before  resizing -----------
    tile_size = (int(image_width / tiles_per_side), int(image_height / tiles_per_side))
    # After resizing -------------
    h_size = int(horizontal_size / tiles_per_side)  # target horizontal size of each tile after resizing
    v_size = int(h_size / row.aspect_ratio)  # target vertical size of each tile after resizing, to maintain aspect ratio

    # Let's make tiles from this -------------------------------------------------
    slide = openslide.OpenSlide(image_path)  # using OpenSlide to get access to the image
    # if debug:
        #print(f'Original image_width {image_width}   image_height {image_height}')
        #print(f'individual tile_size before resizing: {tile_size}')
    tiles = []
    big_tile_number = 1
    for v in tqdm(range(0, image_height - tile_size[1] + 1, tile_size[1])):  # The +1 is just to manage the last step in the range
        for h in range(0, image_width - tile_size[0] + 1, tile_size[0]):
            #if debug: print('processing big tile', big_tile_number)
            image = slide.read_region((h, v), 0,
                                      tile_size)  # reading a tile_size area of the image, starting at h,v,position.
            image = np.array(image)
            image = cv2.resize(image, dsize=(h_size, v_size),
                               interpolation=cv2.INTER_NEAREST)  # INTER_CUBIC takes far longer
            tiles.append(image)
            big_tile_number += 1
            #if debug: print('Tile shape:', image.shape)

    # Showing off the tiles in a grid structure ----------------------------------
    if show:
        #print('showing tiles')
        fig, ax = plt.subplots(nrows=tiles_per_side, ncols=tiles_per_side, figsize=(6, 6 / row.aspect_ratio))
        for i, t in enumerate(tiles):
            x_grid = int(i / tiles_per_side)
            y_grid = i % tiles_per_side
            #plt.clf()
            ax[x_grid, y_grid].imshow()
            ax[x_grid, y_grid].axis('off')
            plt.pause(3)
        fig.tight_layout()
        #plt.show()

    # Stitching it all up --------------------------------------------------------
    stitched = np.array(Image.new('RGBA', (h_size * tiles_per_side, v_size * tiles_per_side)))
    if debug:
        print('Beginning the stitching process...')
        #print('First, a placeholder image of shape', stitched.shape)
    for pos, individual_tile in enumerate(tiles):
        x_grid = pos % tiles_per_side
        y_grid = int(pos / tiles_per_side)
        #if debug: print(f'GRID POSITIONS: {x_grid}, {y_grid}')
        stitched[y_grid * v_size:y_grid * v_size + v_size, x_grid * h_size:x_grid * h_size + h_size,:] = individual_tile
        #Showing off the stitching process -----------------------------------------
        #if show:
            #plt.imshow(stitched)
            #plt.show(block = True)

            #demarc()
    #All done ------------------------------------------------------------------
    return stitched

    ################################################


#The largest image is 329 - tackling the beast now
# row = train_csv.loc[329]
# print('=' * 100)
# stitched = tile_resize_stich(row, tiles_per_side=5, show=True, debug=True)
# print('=' * 100)
# plt.imshow(stitched)
# plt.show()

# new_images = []
# for image in range(len(train_csv)):
#   row = train_csv.loc[image]
#   stitchedImage = tile_resize_stich(row, tiles_per_side=5, show=True, debug=True)
#   print(stitchedImage.shape)
#   stitchedImage = stitchedImage[:,:,:3]
#   #stitchedImage = cv2.imread(stitchedImage)
#   B, G, R = cv2.split(stitchedImage)
#   plt.imshow(B)
#   #plt.show()
#   #print(stitchedImage)
#   new_images.append(stitchedImage)

new_images = []
# for image in range(len(train_csv)):

for image in range(1, 10):
    row = train_csv.loc[image]
    stitchedImage = tile_resize_stich(row, tiles_per_side=5, show=False, debug=True)

    # stitchedImage --> (4000,4000,3)
    stitchedImage = stitchedImage[:, :, :3]
    label = train_csv.loc[image]['label']
    im = Image.fromarray(stitchedImage)
    if label == 'CE':
        im.save(f'processed_data/CE/image_{image}.png')
    else:
        im.save(f'processed_data/LAA/image_{image}.png')

#print(new_images)

