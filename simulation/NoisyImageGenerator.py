import argparse
from pathlib import Path

import mrcfile
import numpy as np
from numpy.typing import NDArray
from tqdm.auto import tqdm


def add_noise(image: NDArray[np.float32], snr: float = 0.1) -> NDArray[np.float32]:
    """
    Add noise to the input image.

    Args:
        image (NDArray[np.float32]): Input image.
        snr (float): Signal-to-noise ratio.

    Returns:
        NDArray[np.float32]: Noisy image.
    """
    sample_var = image.var()
    target_noise_var = sample_var / snr
    print(
        f"Sample Variance: {sample_var:.7f}. Target Noise Variance: {target_noise_var:.7f}")
    noise = np.random.normal(0, np.sqrt(target_noise_var), size=image.shape)
    return (image + noise).astype(np.float32)


def main(sim_img_dir: str, output_dir: str, snr: float = 0.1) -> None:
    """
    Main function to add noise to simulated micrographs.

    Args:
        sim_img_dir (str): Directory of simulated micrograph mrcfiles.
        output_dir (str): Directory where noisy image would be saved.
        snr (float): Signal-to-noise ratio.
    """
    noise_image_dir = Path(output_dir)
    for path in tqdm(list(Path(sim_img_dir).glob("**/*.mrc"))):
        with mrcfile.open(path) as mrc:
            image = mrc.data.astype(np.float32)
        noise_image = add_noise(image, snr)
        noise_image_path = noise_image_dir.joinpath(
            path.parent.stem, f"{path.stem}.npy")
        np.save(noise_image_path, noise_image)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('simimgdir', type=str,
                        help="Directory of simulated micrograph mrcfiles")
    parser.add_argument('outputdir', type=str,
                        help="Directory where noisy image would be saved")
    parser.add_argument('-snr', type=float, help="Signal-to-noise ratio")
    args = parser.parse_args()
    main(args.simimgdir, args.outputdir, args.snr)
