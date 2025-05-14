<div align="center">
    <img src="assets/banner.png" width="50%" alt="Orthogonium" align="center" />
</div>
<br>


<div align="center">
    <a href="#">
        <img src="https://img.shields.io/badge/Python-3.9+-efefef">
    </a>
    <a href="#">
        <img src="https://img.shields.io/badge/Pytorch-2.0+-00008b">
    </a>
    <a href="https://github.com/deel-ai/orthogonium/actions/workflows/linters.yml">
        <img alt="PyLint" src="https://github.com/deel-ai/orthogonium/actions/workflows/linters.yml/badge.svg">
    </a>
    <a href='https://coveralls.io/github/deel-ai/orthogonium?branch=main'>
        <img src='https://coveralls.io/repos/github/deel-ai/orthogonium/badge.svg?branch=main&kill_cache=1' alt='Coverage Status' />
    </a>
    <a href="https://github.com/deel-ai/orthogonium/actions/workflows/tests.yml">
        <img alt="Tests" src="https://github.com/deel-ai/orthogonium/actions/workflows/tests.yml/badge.svg">
    </a>
    <a href="https://github.com/deel-ai/orthogonium/actions/workflows/python-publish.yml">
        <img alt="Pypi" src="https://github.com/deel-ai/orthogonium/actions/workflows/python-publish.yml/badge.svg">
    </a>
    <a href="https://pepy.tech/project/orthogonium">
        <img alt="Pepy" src="https://static.pepy.tech/badge/orthogonium">
    </a>
    <a href="#">
        <img src="https://img.shields.io/badge/License-MIT-efefef">
    </a>
    <a href="https://arxiv.org/abs/2501.07930">
        <img src="https://img.shields.io/badge/arXiv-2501.07930-b31b1b.svg">
    <a href="https://deel-ai.github.io/orthogonium/">
        <img alt="Documentation" src="https://img.shields.io/badge/Docs-here-0000ff">
    </a>
</div>
<br>

# ✨ Orthogonium: Improved implementations of orthogonal layers

This library aims to centralize, standardize and improve methods to 
build orthogonal layers, with a focus on convolutional layers . We noticed that a layer's implementation play a
significant role in the final performance : a more efficient implementation 
allows larger networks and more training steps within the same compute 
budget. So our implementation differs from original papers in order to 
be faster, to consume less memory or be more flexible. Feel free to read the [documentation](https://deel-ai.github.io/orthogonium/)!

# 📃 What is included in this library ?

| Layer name          | Description                                                                                                                        | Orthogonal ? | Usage                                                                                                                              | Status         |
|---------------------|------------------------------------------------------------------------------------------------------------------------------------|--------------|------------------------------------------------------------------------------------------------------------------------------------|----------------|
| AOC (Adaptive-BCOP) | The most scalable method to build orthogonal convolution. Allows control of kernel size, stride, groups dilation and convtranspose | Orthogonal   | A flexible method for complex architectures. Preserve orthogonality and works on large scale images.                               | done           |
| Adaptive-SC-Fac     | Same as previous layer but based on SC-Fac instead of BCOP, which claims a complete parametrization of separable convolutions      | Orthogonal   | Same as above                                                                                                                      | pending        |
| Adaptive-SOC        | SOC modified to be: i) faster and memory efficient ii) handle stride, groups, dilation & convtranspose                             | Orthogonal   | Good for depthwise convolutions and cases where control over the kernel size is not required                                       | in progress    |
| SLL                 | The original SLL layer, which is already quite efficient.                                                                          | 1-Lipschitz  | Well suited for residual blocks, it also contains ReLU activations.                                                                | done           |
| SLL-AOC             | SLL-AOC is to the downsampling block what SLL is to the residual block (see ResNet paper)                                          | 1-Lipschitz  | Allows to construct a "strided" residual block than can change the number of channels. It adds a convolution in the residual path. | done           |
| Sandwish-AOC        | Sandwish convolutions that uses AOC to replace the FFT. Allowing it to scale to large images.                                      | 1-Lipschitz  |                                                                                                                                    | pending        |
| Adaptive-ECO        | ECO modified to i) handle stride, groups & convtranspose                                                                           | Orthogonal   |                                                                                                                                    | (low priority) |

## directory structure

```
orthogonium
├── layers
│   ├── conv
│   │   ├── AOC
│   │   │   ├── ortho_conv.py # contains AdaptiveOrthoConv2d layer
│   │   ├── AdaptiveSOC
│   │   │   ├── ortho_conv.py # contains AdaptiveSOCConv2d layer (untested)
│   │   ├── SLL
│   │   │   ├── sll_layer.py # contains SDPBasedLipschitzConv, SDPBasedLipschitzDense, SLLxAOCLipschitzResBlock
│   ├── legacy
│   │   ├── original code of BCOP, SOC, Cayley etc.
│   ├── linear
│   │   ├── ortho_linear.py # contains OrthoLinear layer (can be used with BB, QR and Exp parametrization)
│   ├── normalization.py # contains Batch centering and Layer centering
│   ├── custom_activations.py # contains custom activations for 1 lipschitz networks
│   ├── channel_shuffle.py # contains channel shuffle layer  
├── model_factory.py # factory function to construct various models for the zoo
├── losses # loss functions, VRA estimation
```

## AOC:

AOC is a method that allows to build orthogonal convolutions with 
an explicit kernel, that support all features like stride, conv transposed,
grouped convolutions and dilation (and all compositions of these parameters). This approach is highly scalable, and can
be applied to problems like Imagenet-1K.

[//]: # (## Adaptive-SC-FAC:)

[//]: # ()
[//]: # (As AOC is built on top of BCOP method, we can construct an equivalent method constructed on top of SC-Fac instead.)

[//]: # (This will allow to compare performance of the two methods given that they have very similar parametrization. &#40;See our )

[//]: # (paper for discussions about the similarities and differences between the two methods&#41;.)

## Adaptive-SOC:

Adaptive-SOC blend the approach of AOC and SOC. It differs from SOC in the way that it is more memory efficient and 
sometimes faster. It also allows to handle stride, groups, dilation and transposed convolutions. Also, our 
implementation uses AOL to normalize the kernel, which is more stable, more efficient and allows a convergence with less 
iterations. However, it does not allow to control the kernel size explicitly as the resulting kernel size is larger 
than the requested kernel size. It is due to the computation to the exponential of a kernel that increases the kernel 
size at each iteration.

## SLL:

SLL is a method that allows to construct small residual blocks with ReLU activations. We kept most to the original 
implementation, and added `SLLxAOCLipschitzResBlock` that construct a down-sampling residual block by fusing SLL with 
$AOC.

## AOL:

AOL is a method that constructs "almost orthogonal" layers. It ensures lipschitzness of the layer while pushing toward 
orthogonality. It is a good alternative when the orthogonality constraint is not necessary, or when the orthogonality
constraint is too expensive to compute.

## more layers are coming soon !

# 🏠 Install the library:

The library is available on pip,so you can install it by running the following command:
```
pip install orthogonium
```

If you wish to deep dive in the code and edit your local version, you can clone the repository and run the following command 
to install it locally:
```
git clone git@github.com:deel-ai/orthogonium.git
pip install -e .
```

## Use the layer:

```python
from orthogonium.layers.conv.AOC import AdaptiveOrthoConv2d, AdaptiveOrthoConvTranspose2d
from orthogonium.reparametrizers import DEFAULT_ORTHO_PARAMS

# use OrthoConv2d with the same params as torch.nn.Conv2d
kernel_size = 3
conv = AdaptiveOrthoConv2d(
    kernel_size=kernel_size,
    in_channels=256,
    out_channels=256,
    stride=2,
    groups=16,
    dilation=2,
    padding_mode="circular",
    ortho_params=DEFAULT_ORTHO_PARAMS,
)
# conv.weight can be assigned to a torch.nn.Conv2d 

# this works similarly for ConvTranspose2d:
conv_transpose = AdaptiveOrthoConvTranspose2d(
    in_channels=256,
    out_channels=256,
    kernel_size=kernel_size,
    stride=2,
    dilation=2,
    groups=16,
    ortho_params=DEFAULT_ORTHO_PARAMS,
)
```

# 🐯 Model Zoo

Stay tuned, a model zoo will be available soon !



# 💥Disclaimer

Given the great quality of the original implementations, orthogonium do not focus on reproducing exactly the results of
the original papers, but rather on providing a more efficient implementation. Some degradations in the final provable 
accuracy may be observed when reproducing the results of the original papers, we consider this acceptable is the gain 
in terms of scalability is worth it. This library aims to provide more scalable and versatile implementations for people who seek to use orthogonal layers 
in a larger scale setting.

# 🔭 Ressources

## 1 Lipschitz CNNs and orthogonal CNNs

- 1-Lipschitz Layers Compared: [github](https://github.com/berndprach/1LipschitzLayersCompared) and [paper](https://berndprach.github.io/publication/1LipschitzLayersCompared)
- BCOP: [github](https://github.com/ColinQiyangLi/LConvNet) and [paper](https://arxiv.org/abs/1911.00937)
- SC-Fac: [paper](https://arxiv.org/abs/2106.09121)
- ECO: [paper](https://openreview.net/forum?id=Zr5W2LSRhD)
- Cayley: [github](https://github.com/locuslab/orthogonal-convolutions) and [paper](https://arxiv.org/abs/2104.07167)
- LOT: [github](https://github.com/AI-secure/Layerwise-Orthogonal-Training) and [paper](https://arxiv.org/abs/2210.11620)
- ProjUNN-T: [github](https://github.com/facebookresearch/projUNN) and [paper](https://arxiv.org/abs/2203.05483)
- SLL: [github](https://github.com/araujoalexandre/Lipschitz-SLL-Networks) and [paper](https://arxiv.org/abs/2303.03169)
- Sandwich: [github](https://github.com/acfr/LBDN) and [paper](https://arxiv.org/abs/2301.11526)
- AOL: [github](https://github.com/berndprach/AOL) and [paper](https://arxiv.org/abs/2208.03160)
- SOC: [github](https://github.com/singlasahil14/SOC) and [paper 1](https://arxiv.org/abs/2105.11417), [paper 2](https://arxiv.org/abs/2211.08453)

## Lipschitz constant evaluation

- [Spectral Norm of Convolutional Layers with Circular and Zero Paddings](https://arxiv.org/abs/2402.00240) 
- [Efficient Bound of Lipschitz Constant for Convolutional Layers by Gram Iteration](https://arxiv.org/abs/2305.16173)
- [github of the two papers](https://github.com/blaisedelattre/lip4conv/tree/main)


# 📖 Citations
If you find this repository useful for your research, please cite:

```
@misc{boissin2025adaptiveorthogonalconvolutionscheme,
      title={An Adaptive Orthogonal Convolution Scheme for Efficient and Flexible CNN Architectures}, 
      author={Thibaut Boissin and Franck Mamalet and Thomas Fel and Agustin Martin Picard and Thomas Massena and Mathieu Serrurier},
      year={2025},
      eprint={2501.07930},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2501.07930}, 
}
```

# 🍻 Contributing

This library is still in a very early stage, so expect some bugs and missing features. Also, before the version 1.0.0,
the API may change and no backward compatibility will be ensured (code is expected to keep working under minor changes
but the loading of parametrized network could fail). This will allow a rapid integration of new features, if you project
to release a trained architecture, exporting the convolutions to torch.nn.conv2D is advised (by saving the `weight` 
attribute of a layer). If you plan to release a training script, fix the version in your requirements.
In order to prioritize the development, we will focus on the most used layers and models. If you have a specific need,
please open an issue, and we will try to address it as soon as possible.

Also, if you have a model that you would like to share, please open a PR with the model and the training script. We will
be happy to include it in the zoo.

If you want to contribute, please open a PR with the new feature or bug fix. We will review it as soon as possible.

## Ongoing developments

Layers:
- SOC:
  - remove channels padding to handle ci != co efficiently
  - enable groups
  - enable support for native stride, transposition and dilation
- AOL:
  - torch implementation of AOL
- Sandwich:
  - import code
  - plug AOC into Sandwich conv

ZOO:
- models from the paper
