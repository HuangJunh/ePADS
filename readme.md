# ePADS
J. Huang, B. Xue, Y. Sun, M. Zhang, and G. G. Yen, “Efficient Perturbation-Aware Distinguishing Score for Zero-Shot Neural Architecture Search,” Applied Soft Computing, 2025.

📑 [Read the Paper]()

## Preparation
This code is tested with Python 3.12.7, PyTorch 2.5.1, and CUDA 12.7. 

- Download datasets (CIFAR-10, CIFAR-100, ImageNet16-120) from https://drive.google.com/drive/folders/1T3UIyZXUhMmIuJLOBMIYKAsJknAtrrO4 and put them in `./datasets`
- Download benchmark datasets and put them in `./APIs`
    - NAS-Bench-201: https://drive.google.com/file/d/16Y0UwGisiouVRxW-W5hEtbxmcHw_0hF_/view
    - NDS: https://dl.fbaipublicfiles.com/nds/data.zip
- Evaluate ePADS on NAS-Bench-201 and NDS by running `sh run.sh`

## Citation
If you use this code in your research, please cite the following paper:
```bibtex
@ARTICLE{ePADS,
  author={Huang, Junhao and Xue, Bing and Sun, Yanan and Zhang, Mengjie and Yen, Gary G.},
  title={Efficient Perturbation-Aware Distinguishing Score for Zero-Shot Neural Architecture Search},
  journal={Applied Soft Computing},
  year={2025},
  volume={},
  number={},
  pages={},
  doi={}}
```
