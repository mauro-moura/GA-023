
from PIL import Image
import numpy as np

from numba import jit, prange

@jit(nopython=True, parallel=True)
def convert_images(ds_full: np.ndarray):
    images = np.empty((len(ds_full), 100, 100), dtype=np.uint8)
    for i in prange(len(ds_full)):
        img = ds_full[i].numpy()
        img = Image.fromarray(img)
        img = img.resize((100, 100))
        img = img.convert('L')
        images[i] = np.array(img)
    
    return images

def load_data(size=100, lim=100, use_jit=False):
    import deeplake

    # Load the FFHQ dataset
    ds = deeplake.load("hub://activeloop/ffhq")

    ds_full = ds['images_1024']['image'][:lim].numpy()

    if use_jit:
        images = convert_images(ds_full)
        return images

    images = np.empty((len(ds_full), size, size), dtype=np.uint8)
    for i in prange(len(ds_full)):
        img = ds_full[i]
        img = Image.fromarray(img)
        img = img.resize((size, size))
        img = img.convert('L')
        images[i] = np.array(img)

    return images
