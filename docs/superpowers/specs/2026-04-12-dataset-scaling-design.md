# Dataset Isotropic Scaling & GAN Feature Alignment Design

## Overview
This design implements a fix for mismatched feature spaces during WGAN-GP training in `train_GAN.py`. By default, the real data was passed through a `SampleDecoder` with a Tanh activation to force it into a `[-1, 1]` range, which was an artificial alignment. Instead, we will geometrically scale the raw aircraft dataset to naturally encourage the VAE encoder to output bounded latent representations, allowing us to drop the artificial transformation.

## 1. Dataset Generation (`data/generate_dataset.py`)
### Component: Isotropic Scaling per Instance
- During aircraft generation, we will compute the global bounding coordinates of the aircraft geometry.
- **Variables scaled**: `[x1, y1, z1, x2, y2, z2, L1, H1, L2, H2]`. The one-hot encoding `[C1, C2, C3]` remains unscaled.
- **Algorithm**: 
  - Find the absolute maximum spatial value (including width/height extents) across all generated boxes for a single aircraft.
  - Scale all continuous dimensions (coords and sizes) by dividing by this maximum value.
  - This maps every aircraft instance to an isotropic bounding space bounded perfectly by `[-1, 1]`.

## 2. GAN Discriminator Alignment (`train_GAN.py`)
### Component: Real Features Generation
- **Removal**: The line `real_features_list.append(decoder.sampleDecoder(root_code))` will be modified.
- **New Behavior**: `real_features_list.append(root_code)`. The `real_features` fed to the Critic/Discriminator will directly be the output from the VAE encoder.
- **Fake Features**: The generator `fake_features = decoder.sampleDecoder(z_p)` remains unmodified. The natural bounding of the scaled dataset will closely map the VAE `root_code` domain to the fake generator domain.

## Testing & Validation
- **Unit Test**: Rerun dataset generation to confirm `generate_dataset.py` produces `[-1, 1]` bounded values in `.mat` output files.
- **Integration**: Rerun the GAN training `train_GAN.py` to ensure the WGAN-GP discriminator converges stably with the un-mapped `root_code` features.
