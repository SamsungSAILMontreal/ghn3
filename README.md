
# Can We Scale Transformers to Predict Parameters of Diverse ImageNet Models?
[Boris Knyazev](http://bknyaz.github.io/), [Doha Hwang](https://mila.quebec/en/person/doha-hwang/), [Simon Lacoste-Julien](http://www.iro.umontreal.ca/~slacoste/)

https://arxiv.org/abs/2303.04143

# Introduction

This work extends the previous work [`Parameter Prediction for Unseen Deep Architectures`](https://github.com/facebookresearch/ppuda/) that introduced improved Graph HyperNetworks (GHN-2).
Here, we scale up GHN-2 and release our best performing model `GHN-3-XL/m16`. 
Our GHN-3 can be used as a good initialization for many large ImageNet models. 

Below are a few figures showcasing our results (see the paper for details).


<figure> <img src="figs/fig1.png" height="380"></figure>

<figure> <img src="figs/fig4.png" height="160"></figure>

<figure> <img src="figs/fig6.png" height="150"></figure>


Using GHN-3 is straightforward to use as shown below with PyTorch examples.


# Installation

```
pip install git+https://github.com/facebookresearch/ppuda.git

pip install torch==1.12.1 torchvision==0.13.1  # our code may work with other versions but not guaranteed

pip install huggingface_hub  # to load the GHN-3 model

```

# Usage


```
import torchvision
from ghn3_utils import from_pretrained

ghn = from_pretrained()

model = torchvision.models.resnet50()  # can be any torchvision model
model = ghn(model, GraphBatch([Graph(model)]))

# That's it, the ResNet-50 is initialized with our GHN-3.

```

GHN-3 is stored in HuggingFace at 
https://huggingface.co/SamsungSAILMontreal/ghn3/tree/main.
As the model takes about 2.5GB, 
it takes a while to download the model during 
the first call of `ghn = from_pretrained()`.

Also see [example.ipynb](example.ipynb) where we show how to predict parameters for all PyTorch models.

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



