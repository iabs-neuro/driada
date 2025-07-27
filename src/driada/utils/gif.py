
import tqdm
import imageio
import matplotlib.pyplot as plt
import numpy as np
import os
from os import listdir, remove
from os.path import isfile, join, splitext, dirname

from .plot import *


def erase_all(path, signature='', ext='.png'):
    """Delete all files in a directory matching signature and extension.
    
    Parameters
    ----------
    path : str
        Directory path to search for files
    signature : str, optional
        String that must be contained in filename (default: '')
    ext : str, optional
        File extension to match, including dot (default: '.png')
    """
    if not os.path.exists(path):
        return
        
    prev_files_paths = [join(path, f) for f in listdir(path) if signature in f]
    files_to_del = [fp for fp in prev_files_paths if isfile(fp) and splitext(fp)[1] == ext]
    for fp in files_to_del:
        remove(fp)


def save_image_series(path, figures, im_ext='png'):
    """Save a series of matplotlib figures to disk.
    
    Parameters
    ----------
    path : str
        Directory path where images will be saved
    figures : list
        List of matplotlib figure objects
    im_ext : str, optional
        Image extension without dot (default: 'png')
    """
    os.makedirs(path, exist_ok=True)
    
    for i in tqdm.tqdm(range(len(figures)), leave=True, position=0):
        fig = figures[i]
        if hasattr(fig, '_suptitle') and fig._suptitle:
            figname = fig._suptitle.get_text() + '.' + im_ext
        else:
            figname = f'figure_{i:04d}.{im_ext}'
        fig.savefig(join(path, figname))
        plt.close(fig)


def create_gif_from_image_series(path, signature, gifname, erase_prev=True, im_ext='png', duration=0.2):
    """Create an animated GIF from a series of images.
    
    Parameters
    ----------
    path : str
        Directory containing the source images
    signature : str
        String that must be contained in image filenames to include
    gifname : str
        Name for the output GIF file (without extension)
    erase_prev : bool, optional
        Whether to delete matching images after creating GIF (default: True)
    im_ext : str, optional
        Image extension to search for (default: 'png')
    duration : float, optional
        Duration of each frame in seconds (default: 0.2)
        
    Returns
    -------
    str
        Path to the created GIF file
    """
    images = []
    imfiles = [f for f in listdir(path) if isfile(join(path, f)) and signature in f and im_ext in f]
    imfiles = sorted(imfiles)

    for filename in tqdm.tqdm(imfiles, leave=True, position=0):
        images.append(imageio.v3.imread((join(path, filename))))

    # Create GIFs directory if it doesn't exist
    gif_dir = join(path, 'GIFs')
    os.makedirs(gif_dir, exist_ok=True)
    
    gif_path = join(gif_dir, f'{signature} {gifname}.gif')
    if images:  # Only create GIF if there are images
        imageio.mimsave(gif_path, images, duration=duration)
    
    # Delete source images after creating GIF if requested
    if erase_prev:
        erase_all(path, signature=signature, ext='.' + im_ext if not im_ext.startswith('.') else im_ext)
    
    return gif_path
