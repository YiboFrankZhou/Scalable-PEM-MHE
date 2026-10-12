# Selected training results

Each Ex2/Ex3 directory contains a pretrained checkpoint and its TensorBoard training record. The train and validation costs refer to the same selected training step.

| Notebook / run | Selected step | Train cost | Validation cost |
|---|---:|---:|---:|
| [Ex2_SUBNET](../Ex2_SUBNET.ipynb) | 292 | 2.853707485e-02 | 1.658165700e-01 |
| [Ex2_Joint-MHE](../Ex2_Joint-MHE.ipynb) | 600 | 1.366484165e-02 | 4.422417376e-03 |
| [Ex2_EKF-PEM](../Ex2_EKF-PEM.ipynb) | 2 | 1.328519457e-04 | 6.489109287e-05 |
| [Ex2_EKF-PEM-adaptive](../Ex2_EKF-PEM-adaptive.ipynb) | 355 | 1.595451322e-04 | 9.522768960e-05 |
| [Ex2_PEM-MHE](../Ex2_PEM-MHE.ipynb) | 3055 | 7.151468822e-07 | 7.932147241e-07 |
| [Ex3_SUBNET](../Ex3_SUBNET.ipynb) | 1064 | 3.163922723e-04 | 3.254300444e-03 |
| [Ex3_Joint-MHE](../Ex3_Joint-MHE.ipynb) | 2000 | 4.704400490e-04 | 4.471864377e-04 |
| [Ex3_EKF-PEM](../Ex3_EKF-PEM.ipynb) | 1 | 4.367503570e-04 | 4.201322736e-04 |
| [Ex3_EKF-PEM-adaptive](../Ex3_EKF-PEM-adaptive.ipynb) | 76 | 2.201880270e-04 | 2.620636078e-04 |
| [Ex3_PEM-MHE-physics](../Ex3_PEM-MHE-physics.ipynb) | 875 | 4.238535475e-04 | 4.102281237e-04 |
| [Ex3_PEM-MHE-graybox](../Ex3_PEM-MHE-graybox.ipynb) | 960 | 2.920935804e-04 | 2.837149368e-04 |

In every Ex2/Ex3 run directory:

- `model_params_opt.pth`: selected learned weights and parameters.
- `metrics.json`: recorded costs and selected step.
- `events.out.tfevents.recorded` and scalar subdirectories: TensorBoard training records.
- `initialization.npz`, where present: experiment initialization; the adaptive notebooks share the corresponding EKF initialization and load their own encoder checkpoint.

`manifest.json` contains source paths and SHA256 checksums. Run `python tests/verify_results.py` from the repository root to verify the saved results.

Load pretrained parameters in the corresponding notebook. To view training history, run `tensorboard --logdir runs` from the repository root.

`Ex1/` contains the recorded timing measurements (`point_results.npz`), benchmark settings (`metadata.json`), and their source checksums (`provenance.json`). The plot notebook reads this archive and writes its PDF and PNG to `results/Ex1/`.

New training and benchmark runs go to `new/<experiment>_<timestamp>/`, which is excluded from Git.
