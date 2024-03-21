# SGENet
[![arXiv](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/2403.13330)

An official Pytorch implement of the paper "Efficient scene text image super-resolution with semantic guidance
" (ICASSP2024).

Authors: *LeoWu TomyEnrique, Xiangcheng Du, Kangliang Liu, Han Yuan, Zhao Zhou and Cheng Jin*


![SGENet-pipeline](https://arxiv.org/html/2403.13330v1/x1.png)

## Samples

![test-samples](https://arxiv.org/html/2403.13330v1/x2.png)

## Pre-trained Model

Coming soon...

## Prepare Datasets

TextZoom dataset contains 17,367 LR-HR pairs for training and 4,373 pairs for testing.

Data (Lmdb): [Badiu NetDisk](https://pan.baidu.com/s/1PYdNqo0GIeamkYHXJmRlDw). password: **kybq**; 
[Google Drive](https://drive.google.com/drive/folders/1WRVy-fC_KrembPkaI68uqQ9wyaptibMh?usp=sharing)

## Text Recognizers

Download the pretrained recognizer from:
```
Aster: https://github.com/ayumiymk/aster.pytorch  
MORAN:  https://github.com/Canjie-Luo/MORAN_v2  
CRNN: https://github.com/meijieru/crnn.pytorch
```

## Citation

```
@INPROCEEDINGS{10446964,
  author={TomyEnrique, LeoWu and Du, Xiangcheng and Liu, Kangliang and Yuan, Han and Zhou, Zhao and Jin, Cheng},
  booktitle={ICASSP 2024 - 2024 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)}, 
  title={Efficient Scene Text Image Super-Resolution with Semantic Guidance}, 
  year={2024},
  volume={},
  number={},
  pages={3160-3164},
  keywords={Image recognition;Text recognition;Computational modeling;Semantics;Superresolution;Feature extraction;Computational efficiency;Scene text image super-resolution;efficient model;semantic guidance},
  doi={10.1109/ICASSP48485.2024.10446964}
}
```



## Acknowledgement

The code of this work is based on [LEMMA](https://github.com/csguoh/LEMMA), [TATT](https://github.com/mjq11302010044/TATT), and [TextZoom](https://github.com/WenjiaWang0312/TextZoom). Thanks for your contributions.
