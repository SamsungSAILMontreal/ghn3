
# Can We Scale Transformers to Predict Parameters of Diverse ImageNet Models?

**To appear at [ICML 2023](https://icml.cc/Conferences/2023)**

[Boris Knyazev](http://bknyaz.github.io/), [Doha Hwang](https://mila.quebec/en/person/doha-hwang/), [Simon Lacoste-Julien](http://www.iro.umontreal.ca/~slacoste/)

https://arxiv.org/abs/2303.04143

# Introduction

**Updates**

- [June 6, 2023] GHN-3 code improved (see [ghn3/nn.py](ghn3/nn.py)), more examples added (see [example_single_model.py](example_single_model.py)).
- [Apr 11, 2023] Cleaned up graph construction, sanity check for all PyTorch models.
- [Apr 4, 2023] Slightly updated graph construction for ViT to be consistent with our paper. 
Made four variants of our GHN-3 available: `ghn3tm8, ghn3sm8, ghn3lm8, ghn3xlm16` (see updated [example.ipynb](example.ipynb)).
`ghn3tm8` takes just 27MB so it is efficient to use in low-memory cases.
 

This work extends the previous work [`Parameter Prediction for Unseen Deep Architectures`](https://github.com/facebookresearch/ppuda/) that introduced improved Graph HyperNetworks (GHN-2).
Here, we scale up GHN-2 and release our best performing model `GHN-3-XL/m16` as well as smaller variants. 
Our GHN-3 can be used as a good initialization for many large ImageNet models. 

Below are a few figures showcasing our results (see [our paper](https://arxiv.org/abs/2303.04143) for details).


<figure> <img src="figs/fig1.png" height="380"></figure>

<figure> <img src="figs/fig4.png" height="160"></figure>

<figure> <img src="figs/fig6.png" height="150"></figure>


Our code has only a few dependencies and is easy to use as shown below with PyTorch examples.

Please feel free to open a GitHub issue to ask questions or report bugs. 
Pull requests are also welcome.

# Installation

```
pip install git+https://github.com/facebookresearch/ppuda.git  # to have core functionality

pip install torch>=1.12.1 torchvision>=0.13.1  # [optional] update torch in case it's old

pip install huggingface_hub  # to load pretrained GHNs

```

# Usage


```
import torch
import torchvision
from ghn3 import from_pretrained

ghn = from_pretrained()  # default is 'ghn3xlm16.pt', other variants are: 'ghn3tm8.pt', 'ghn3sm8.pt', 'ghn3lm8.pt'

model = torchvision.models.resnet50()  # can be any torchvision model
model = ghn(model)

# That's it, the ResNet-50 is initialized with our GHN-3.
```


GHN-3 is stored in HuggingFace at 
https://huggingface.co/SamsungSAILMontreal/ghn3/tree/main.
As the largest model (`ghn3xlm16.pt`) takes about 2.5GB, 
it takes a while to download the model during 
the first call of `ghn = from_pretrained()`.

See [example_single_model.py](example_single_model.py) for the examples of fine-tuning the model or GHN.
Also see [example_all_pytorch.ipynb](example_all_pytorch.ipynb) where we show how to predict parameters for all PyTorch models.

# License

This code is licensed under [MIT license](LICENSE) and is based on
https://github.com/facebookresearch/ppuda that is also licensed under [MIT license](https://github.com/facebookresearch/ppuda/blob/main/LICENSE).

# Citation

```
@article{knyazev2023canwescale,
  title={Can We Scale Transformers to Predict Parameters of Diverse ImageNet Models?},
  author={Knyazev, Boris and Hwang, Doha and Lacoste-Julien, Simon},
  journal={arXiv preprint arXiv:2303.04143},
  year={2023}
}
```



