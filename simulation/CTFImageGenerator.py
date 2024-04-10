import argparse
from typing import List

import numpy as np
import pandas as pd
import starfile
from aspire.source import MicrographSimulation
from aspire.volume import Volume
from aspire.operators import RadialCTFFilter


def get_ctfs(refine_result_path: str, pixel_size: float, voltage: float, spherical_aberration: float, amplitude_contrast: float) -> List[RadialCTFFilter]:
    """
    Get CTF filters from refinement results.

    Args:
        refine_result_path (str): Path to refinement results.
        pixel_size (float): Pixel size in Angstroms.
        voltage (float): Voltage in kV.
        spherical_aberration (float): Spherical aberration in mm.
        amplitude_contrast (float): Amplitude contrast.

    Returns:
        List[RadialCTFFilter]: List of RadialCTFFilter objects.
    """
    refine_result: pd.DataFrame = starfile.read(refine_result_path)
    n_micrographs: int = len(refine_result.rlnImageName.unique())
    random_defocus_values: pd.Series = refine_result.rlnDefocusV.sample(
        n_micrographs, random_state=42)
    return [RadialCTFFilter(pixel_size=pixel_size, voltage=voltage, defocus=d, Cs=spherical_aberration, alpha=amplitude_contrast, B=0)
            for d in random_defocus_values]


def get_volume(vol_path: str) -> Volume:
    """
    Get Volume object from volume path.

    Args:
        vol_path (str): Path to volume.

    Returns:
        Volume: Volume object.
    """
    return Volume.load(vol_path, dtype=np.float64)


def main(vol_path: str, refine_result_path: str, output_dir: str, pixel_size: float, voltage: float, spherical_aberration: float, amplitude_contrast: float, particles_per_micrograph: int = 1, seed: int = None) -> None:
    """
    Main function for micrograph simulation.

    Args:
        vol_path (str): Path to volume.
        refine_result_path (str): Path to refinement results.
        output_dir (str): Output directory.
        particles_per_micrograph (int): Number of particles per micrograph. Default is 1.
        seed (int): Random seed. Default is None.
        pixel_size (float): Pixel size in Angstroms. Default is 1.77.
        voltage (float): Voltage in kV.
        spherical_aberration (float): Spherical aberration in mm.
        amplitude_contrast (float): Amplitude contrast.
    """
    vol = get_volume(vol_path)
    ctfs = get_ctfs(refine_result_path, pixel_size, voltage,
                    spherical_aberration, amplitude_contrast)

    src = MicrographSimulation(
        volume=vol,
        micrograph_size=4096,
        micrograph_count=len(ctfs),
        particles_per_micrograph=particles_per_micrograph,
        particle_amplitudes=1,
        seed=seed,
        ctf_filters=ctfs,
    )
    src.save(output_dir, name_prefix='sim_image')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('vol_path', type=str, help="The path of volume")
    parser.add_argument('refine_result_path', type=str,
                        help="The path of result of 3d refinement")
    parser.add_argument('outputdir', type=str,
                        help="Directory where noisy image would be saved")
    parser.add_argument('-n_particle', type=int, default=150,
                        help="Number of particles per micrograph")
    parser.add_argument('--seed', type=int, default=42, help="Random seed")
    parser.add_argument('--pixel_size', type=float,
                        default=1.77, help="Pixel size in Angstroms")
    parser.add_argument('--voltage', type=float,
                        default=300, help="Voltage in kV")
    parser.add_argument('--spherical_aberration', type=float,
                        default=2.7, help="Spherical aberration in mm")
    parser.add_argument('--amplitude_contrast', type=float,
                        default=0.1, help="Amplitude contrast")
    args = parser.parse_args()
    main(args.vol_path, args.refine_result_path, args.outputdir, args.pixel_size, args.voltage,
         args.spherical_aberration, args.amplitude_contrast, particles_per_micrograph=args.n_particle, seed=args.seed)
    # refine_result_path = "/content/drive/MyDrive/selected.star"
    # vol_path = "/content/drive/MyDrive/J3_003_volume_map.mrc"
    # vol_path = "/content/drive/MyDrive/J3_003_volume_mask_refine.mrc"
