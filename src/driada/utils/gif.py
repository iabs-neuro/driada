import tqdm
import imageio
import matplotlib.pyplot as plt
import os
from os import listdir, remove
from os.path import isfile, join, splitext


def erase_all(path, signature="", ext=".png"):
    """Delete all files in a directory matching signature and extension.
    
    Searches for files in the specified directory that contain the given
    signature string in their filename and have the specified extension,
    then deletes them. If the directory doesn't exist, returns silently.

    Parameters
    ----------
    path : str
        Directory path to search for files.
    signature : str, optional
        String that must be contained in filename (default: '').
    ext : str, optional
        File extension to match, including dot (default: '.png').
        
    Raises
    ------
    OSError
        If file deletion fails due to permissions or file being in use.
        
    Notes
    -----
    This function is typically used to clean up temporary image files
    before creating new visualizations. It will not raise an error if
    the directory doesn't exist.
    
    Examples
    --------
    >>> # Delete all PNG files in a directory
    >>> erase_all('/tmp/images', ext='.png')
    
    >>> # Delete only files containing 'temp' in the name
    >>> erase_all('/tmp/images', signature='temp', ext='.jpg')
    
    See Also
    --------
    ~driada.utils.gif.save_image_series :
        Save multiple figures to disk.
    """
    if not os.path.exists(path):
        return

    prev_files_paths = [join(path, f) for f in listdir(path) if signature in f]
    files_to_del = [
        fp for fp in prev_files_paths if isfile(fp) and splitext(fp)[1] == ext
    ]
    for fp in files_to_del:
        remove(fp)


def save_image_series(path, figures, im_ext="png"):
    """Save a series of matplotlib figures to disk.
    
    Saves each figure in the provided list to the specified directory.
    Creates the directory if it doesn't exist. Figures are named using
    their suptitle if available, otherwise using a numbered sequence.
    Each figure is closed after saving to free memory.

    Parameters
    ----------
    path : str
        Directory path where images will be saved.
    figures : list
        List of matplotlib figure objects.
    im_ext : str, optional
        Image extension without dot (default: 'png').
        
    Raises
    ------
    OSError
        If directory creation fails or file cannot be saved.
    AttributeError
        If an element in figures list is not a valid matplotlib figure.
        
    Notes
    -----
    This function displays a progress bar using tqdm while saving figures.
    All figures are automatically closed after saving to prevent memory leaks.
    If a figure has a suptitle, it will be used as the filename.
    
    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> # Create multiple figures
    >>> figs = []
    >>> for i in range(5):
    ...     fig, ax = plt.subplots()
    ...     _ = ax.plot([1, 2, 3], [i, i+1, i+2])
    ...     _ = fig.suptitle(f'Plot_{i}')
    ...     figs.append(fig)
    >>> # save_image_series('/tmp/plots', figs, im_ext='png')
    
    See Also
    --------
    ~driada.utils.gif.create_gif_from_image_series :
        Create animated GIF from saved images.
    ~driada.utils.gif.erase_all :
        Clean up image files.
    """
    os.makedirs(path, exist_ok=True)

    for i in tqdm.tqdm(range(len(figures)), leave=True, position=0):
        fig = figures[i]
        if hasattr(fig, "_suptitle") and fig._suptitle:
            figname = fig._suptitle.get_text() + "." + im_ext
        else:
            figname = f"figure_{i:04d}.{im_ext}"
        fig.savefig(join(path, figname))
        plt.close(fig)


def create_gif_from_image_series(
    path, signature, gifname, erase_prev=True, im_ext="png", duration=0.2
):
    """Create an animated GIF from a series of images.
    
    Searches for images in the specified directory that contain the signature
    string in their filename, sorts them alphabetically, and combines them
    into an animated GIF. The GIF is saved in a 'GIFs' subdirectory which
    is created automatically if it doesn't exist.

    Parameters
    ----------
    path : str
        Directory containing the source images.
    signature : str
        String that must be contained in image filenames to include.
    gifname : str
        Name for the output GIF file (without extension).
    erase_prev : bool, optional
        Whether to delete matching images after creating GIF (default: True).
    im_ext : str, optional
        Image extension to search for (default: 'png').
    duration : float, optional
        Duration of each frame in seconds (default: 0.2).

    Returns
    -------
    str
        Path to the created GIF file.
        
    Raises
    ------
    OSError
        If directory operations fail or images cannot be read/written.
    ValueError
        If no matching images are found (from imageio).
        
    Notes
    -----
    The function creates a 'GIFs' subdirectory within the input path to store
    the output GIF. Images are sorted alphabetically by filename before being
    added to the GIF, so proper naming (e.g., frame_0001.png, frame_0002.png)
    ensures correct order. A progress bar shows the image loading process.
    
    The function handles image extensions flexibly - 'png' and '.png' are
    treated the same way.
    
    Examples
    --------
    Create GIF from all PNG images containing 'frame' in the name::
    
        gif_path = create_gif_from_image_series(
            '/tmp/images', 
            signature='frame',
            gifname='animation',
            duration=0.5
        )
    
    Keep source images after creating GIF::
    
        gif_path = create_gif_from_image_series(
            '/tmp/images',
            signature='plot_',
            gifname='results',
            erase_prev=False
        )
    
    See Also
    --------
    ~driada.utils.gif.save_image_series :
        Save matplotlib figures as image series.
    ~driada.utils.gif.erase_all :
        Delete files matching specific criteria.
    """
    images = []
    imfiles = [
        f
        for f in listdir(path)
        if isfile(join(path, f)) and signature in f and im_ext in f
    ]
    imfiles = sorted(imfiles)

    for filename in tqdm.tqdm(imfiles, leave=True, position=0):
        images.append(imageio.v3.imread((join(path, filename))))

    # Create GIFs directory if it doesn't exist
    gif_dir = join(path, "GIFs")
    os.makedirs(gif_dir, exist_ok=True)

    gif_path = join(gif_dir, f"{signature} {gifname}.gif")
    if images:  # Only create GIF if there are images
        imageio.mimsave(gif_path, images, duration=duration)

    # Delete source images after creating GIF if requested
    if erase_prev:
        erase_all(
            path,
            signature=signature,
            ext="." + im_ext if not im_ext.startswith(".") else im_ext,
        )

    return gif_path
