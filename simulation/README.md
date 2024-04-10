# Micrograph Simulation

## Package Require:
- mrcfile
- starfile
- scikit-image
- aspire

## Usage:

### Generate Simulated Micrograph without Noise

To generate simulated micrograph without noise, use the following code:

```bash
python NoisyImageGenerator.py volume_path refine_result_path simulated_micrograph_directory
```

This command will generate simulated micrographs without noise in the specified directory.

Directory structure:

```bash
simulated_micrograph_directory/
├── sim_image_0.mrc
├── sim_image_0.star
├── sim_image_1.mrc
├── sim_image_1.star
│   ...
├── sim_image_83.mrc
└── sim_image_83.star
```

### Split the Dataset

The generated simulated micrographs should be split into train-validation-test manually:

Directory structure:

```bash
simulated_micrograph_directory/
├── train/
│   ├── sim_image_0.mrc
│   ├── sim_image_0.star
│   │   ...
│   ├── sim_image_57.mrc
│   └── sim_image_57.star
├── val/
│   ├── sim_image_58.mrc
│   ├── sim_image_58.star
│   │   ...
│   ├── sim_image_66.mrc
│   └── sim_image_66.star
└── test/
    ├── sim_image_67.mrc
    ├── sim_image_67.star
    │   ...
    ├── sim_image_83.mrc
    └── sim_image_83.star
```

### Generate Simulated Micrograph with Noise

To generate simulated micrograph with noise under given SNR (default: 0.1), use the following code:

```bash
python NoisyImageGenerator.py simulated_micrograph_directory noisy_image_directory -snr 0.1
```

This command will generate simulated micrographs with noise in the specified directory.

Directory structure:

```bash
noisy_image_directory/
├── train/
│   ├── sim_image_0.mrc
│   ├── sim_image_0.star
│   │   ...
│   ├── sim_image_57.mrc
│   └── sim_image_57.star
├── val/
│   ├── sim_image_58.mrc
│   ├── sim_image_58.star
│   │   ...
│   ├── sim_image_66.mrc
│   └── sim_image_66.star
└── test/
    ├── sim_image_67.mrc
    ├── sim_image_67.star
    │   ...
    ├── sim_image_83.mrc
    └── sim_image_83.star
```

### Generate Ground Truth

To generate ground truth of the simulated micrographs, use the following code:

```bash
python GroundTruthGenerator.py simulated_micrograph_directory ground_truth_directory
```

This command will generate ground truth of the simulated micrographs in the specified directory.

Directory structure:

```bash
noisy_image_directory/
├── sim_image_0.mrc
├── sim_image_0.star
│   ...
├── sim_image_83.mrc
└── sim_image_83.star
```
