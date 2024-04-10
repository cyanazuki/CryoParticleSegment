import argparse
import gc
from pathlib import Path

import mrcfile
import numpy as np
from numpy.typing import NDArray
from PIL import Image
from skimage import morphology
from skimage.filters import threshold_li
from tqdm.auto import tqdm


def threshold(image: NDArray[np.float32]) -> NDArray[np.uint8]:
    """
    Threshold the input image.

    Args:
        image (NDArray[np.float32]): Input image.

    Returns:
        NDArray[np.uint8]: Thresholded image.
    """
    image_threshold = 0.015
    # image_threshold: np.float32 = threshold_li(image)
    return (image > image_threshold).astype(np.uint8) * 255


def processing(image: NDArray[np.float32]) -> NDArray[np.uint8]:
    """
    Apply processing steps to the input image.

    Args:
        image (NDArray[np.float32]): Input image.

    Returns:
        NDArray[np.uint8]: Processed image.
    """
    image = morphology.black_tophat(image, morphology.disk(20))
    image = morphology.remove_small_objects(
        image > 0, min_size=128, connectivity=1)
    image = morphology.closing(image, morphology.disk(20))
    return image.astype(np.uint8) * 255


def numpy_to_PIL(image: NDArray) -> Image:
    """
    Convert numpy array to PIL Image.

    Args:
        image (NDArray): Input image.

    Returns:
        Image: PIL Image.
    """
    return Image.fromarray(image).convert('L')


def self2binary(image: NDArray[np.float_]) -> NDArray[np.uint8]:
    """
    Convert image to binary.

    Args:
        image (NDArray[np.float64]): Input image.

    Returns:
        NDArray[np.uint8]: Binary image.
    """
    clean_image = threshold(image)
    clean_image = processing(clean_image)
    return clean_image


def main(sim_img_dir: str, output_dir: str) -> None:
    """
    Main function to process simulated micrographs.

    Args:
        sim_img_dir (str): Directory of simulated micrograph mrcfiles.
        output_dir (str): Directory where groundtruth would be saved.
    """
    clean_image_dir = Path(output_dir)
    for path in tqdm(list(Path(sim_img_dir).glob("**/*.mrc"))):
        with mrcfile.open(path) as mrc:
            image = mrc.data.astype(np.float32)
        clean_image = self2binary(image[0])
        clean_image_path = clean_image_dir.joinpath(f"{path.stem}.png")
        numpy_to_PIL(clean_image).save(clean_image_path)
        gc.collect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('simimgdir', type=str,
                        help="Directory of simulated micrograph mrcfiles")
    parser.add_argument('outputdir', type=str,
                        help="Directory where groundtruth would be saved")
    args = parser.parse_args()
    main(args.simimgdir, args.outputdir)
