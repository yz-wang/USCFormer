## USCFormer: Unified Transformer With Semantically Contrastive Learning for Image Dehazing (TITS'2023)

Authors: Yongzhen Wang, Jiamei Xiong, Xuefeng Yan and Mingqiang Wei

[[Paper Link]](https://ieeexplore.ieee.org/document/10143384)

### Abstract

Haze severely degrades the visibility of scene objects and deteriorates the performance of autonomous driving, traffic monitoring, and other vision-based intelligent transportation systems. As a potential remedy, we propose a novel unified Transformer with semantically contrastive learning for image dehazing, dubbed USCFormer. USCFormer has three key contributions. First, USCFormer absorbs the respective strengths of CNN and Transformer by incorporating them into a unified Transformer format. Thus, it allows the simultaneous capture of global-local dependency features for better image dehazing. Second, by casting clean/hazy images as the positive/negative samples, the contrastive constraint encourages the restored image to be closer to the ground-truth images (positives) and away from the hazy ones (negatives). Third, we regard the semantic information as important prior knowledge to help USCFormer mitigate the effects of haze on the scene and preserve image details and colors by leveraging intra-object semantic correlation. Experiments on synthetic datasets and real-world hazy photos fully validate the superiority of USCFormer in both perceptual quality assessment and subjective evaluation. Code is available at https://github.com/yz-wang/USCFormer.

#### If you find the resource useful, please cite the following :- )

```
@article{Wang_2023_TITS,
author={Wang, Yongzhen and Xiong, Jiamei and Yan, Xuefeng and Wei, Mingqiang},
  journal={IEEE Transactions on Intelligent Transportation Systems}, 
  title={USCFormer: Unified Transformer With Semantically Contrastive Learning for Image Dehazing}, 
  year={2023},
  volume={},
  number={},
  pages={1-13},
  doi={10.1109/TITS.2023.3277709}}
```  

## Prerequisites:
Python 3.7 or above

Pytorch 1.5

CUDA 10.1
