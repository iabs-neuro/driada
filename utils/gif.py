
import tqdm
import imageio
import matplotlib.pyplot as plt
import numpy as np
from os import listdir, remove
from os.path import isfile, join, splitext

from .plot import *


def erase_all(path, signature='', ext='.png'):
    # destroy all previous images
    prev_files_paths = [join(path, f) for f in listdir(path) if signature in f]
    files_to_del = [fp for fp in prev_files_paths if isfile(fp) and splitext(fp) == ext]
    for fp in files_to_del:
        remove(fp)


def save_image_series(path, figures, im_ext='png'):
    #with io.capture_output() as captured:

    for i in tqdm.tqdm(np.arange(1, len(figures)), leave = True, position = 0):
        fig = figures[i]
        figname = fig._suptitle.get_text() + im_ext
        fig.savefig(join(path, figname))
        fig.close()


def create_gif_from_image_series(path, signature, gifname, erase_prev=True, im_ext='png', duration=0.2):
    if erase_prev:
        erase_all(path, signature=signature, ext=im_ext)

    images = []
    imfiles = [f for f in listdir(path) if isfile(join(path, f)) and signature in f and im_ext in f]
    imfiles = sorted(imfiles)

    for filename in tqdm.tqdm(imfiles, leave=True, position=0):
        images.append(imageio.v3.imread((join(path, filename))))

    imageio.mimsave(join(path, 'GIFs', f'{gifname}.gif'), images, duration=duration)
