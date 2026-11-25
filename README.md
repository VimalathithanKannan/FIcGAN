# FIcGAN
Code and Trained Models for "Pressure-Dependent Prediction of Contact Patches in Rough Elastic Contacts Using Conditional Generative Adversarial Networks"

# Pressure-Dependent Prediction of Contact Patches in Rough Elastic Contacts Using Conditional Generative Adversarial Networks
*Manuscript in preparation.*

## Repository contents

| File | Description |
|------|-------------|
| `cGAN_Training_Evaluation.py` | Training and evaluation of the conditional GAN (cGAN) |
| `FIcGAN_Training_Evaluation.py` | Training and evaluation of the FIcGAN model (feature-informed cGAN) |
| `trained_cGAN_Model.h5` | Trained cGAN model (Keras HDF5 format) |
| `trained_FIcGAN_Model.h5` | Trained FIcGAN model (Keras HDF5 format) |
| `requirements.txt` | Python dependencies with the versions used |

## Installation

Python 3.9.18 was used. To install the dependencies:

```bash
pip install -r requirements.txt
```

## Data

The data used in the manuscript will be made available upon request. To run the scripts on your own data, the files must be in **`.dat` format**:

- Input data - Rough surface height maps and imposed pressure; Output data - Binarized or non-binarized contact maps
- Data size used to train the models is 128x128; but larger or smaller size can be used. Adjust the block_size and range in create_dataset function to scale the network for your requirements.

## Usage

### Training and evaluation

Edit the data path wherever applicable, then run:

```bash
python cGAN_Training_Evaluation.py
python FIcGAN_Training_Evaluation.py
```

### Using the trained models

```python
from tensorflow.keras.models import load_model

cgan = load_model("trained_cGAN_Model.h5")
ficgan = load_model("trained_FIcGAN_Model.h5")
```

The model requires local surface height map and imposed pressure scalar as inputs and it will predict and write the contact map on the predefined filepath.

## Citation

If you use this code or the trained models, please cite the manuscript:

```
VP Kannan et al. Pressure-Dependent Prediction of Contact Patches in Rough Elastic Contacts Using Conditional Generative Adversarial Networks. [2026, manuscript in preparation].
```

This section will be updated with the journal and DOI after publication.

Codes and trained moels archived on Zenodo:
https://doi.org/10.5281/zenodo.24538868

## License

Released under the [MIT License](LICENSE).

## Contact

Vimalthithan Paramsamy Kannan
vimal.p.k@hotmail.com
pk.vimalathithan@poliba.it
