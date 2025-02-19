import torch.nn as nn
from torch.nn import AvgPool2d

from orthogonium.layers.residual import PrescaledAdditiveResidual
from orthogonium.model_factory.classparam import ClassParam
from orthogonium.layers import AdaptiveOrthoConv2d
from orthogonium.layers import BatchCentering2D
from orthogonium.layers import LayerCentering2D
from orthogonium.layers import MaxMin
from orthogonium.layers import OrthoLinear
from orthogonium.layers import UnitNormLinear
from orthogonium.layers.conv.SLL.sll_layer import SLLxAOCLipschitzResBlock
from orthogonium.layers.conv.SLL.sll_layer import SDPBasedLipschitzResBlock
from orthogonium.reparametrizers import DEFAULT_ORTHO_PARAMS

from orthogonium import layers as ol

def SLLxBCOPResNet50(
    img_shape=(3, 224, 224),
    n_classes=1000,
    norm=None,  # ClassParam(BatchCentering2D),
    # norm=ClassParam(LayerCentering2D),
    # pool=ClassParam(nn.LPPool2d, norm_type=2),
):
    act = MaxMin  # torch.nn.ReLU
    # act2 = Abs, MaxMin, ReLU
    layers = [
        # conv2d stride 2 + norm + act
        AdaptiveOrthoConv2d(
            in_channels=img_shape[0],
            out_channels=64,
            kernel_size=7,
            stride=2,
            padding=3,
        ),
        act(),
        # SDPBasedLipschitzResBlock(
        #     cin=3,
        #     cout=64,
        #     inner_dim_factor=0.25,
        #     kernel_size=7,
        #     stride=2,
        # ),
        norm(64) if norm is not None else nn.Identity(),
        # max pool ks 3 stride 2
        # opt 1: replace max pool with l2 pool
        # opt 2: replace max pool with strided conv
        # SDPBasedLipschitzResBlock(
        #     cin=64,
        #     cout=64,
        #     inner_dim_factor=0.25,
        #     kernel_size=7,
        #     stride=2,
        # ),
        # nn.AvgPool2d(2, stride=2, padding=0, divisor_override=2),
        nn.LPPool2d(kernel_size=2, norm_type=2),
        # OrthoConv2d(
        #     in_channels=64,
        #     out_channels=64,
        #     kernel_size=3,
        #     stride=2,
        #     padding=1,
        # ),
        # act(),
        norm(64) if norm is not None else nn.Identity(),
        *ResNet50Block(64, 256, 3, norm, act, stride=1),
        *ResNet50Block(256, 512, 4, norm, act, stride=2),
        *ResNet50Block(512, 1024, 6, norm, act, stride=2),
        *ResNet50Block(1024, 2048, 3, norm, act, stride=2),
        nn.LPPool2d(kernel_size=7, norm_type=2),
        # nn.AvgPool2d(7, divisor_override=7),
        # OrthoConv2d(
        #     in_channels=2048,
        #     out_channels=2048,
        #     kernel_size=7,
        #     stride=7,
        #     padding=0,
        # ),
        # norm() if norm is not None else nn.Identity(),
        nn.Flatten(),
        # OrthoLinear(2048, 2048),
        # norm() if norm is not None else nn.Identity(),
        # act2(),
        OrthoLinear(2048, n_classes, bias=True),
    ]
    return nn.Sequential(*layers)


def ResNet50Block(in_channels, out_channels, n_blocks, norm, act, stride=2):
    layers = []
    layers.append(
        SDPBasedLipschitzResBlock(
            cin=in_channels,
            cout=out_channels,
            inner_dim_factor=1.0,
            kernel_size=3,
            stride=stride,
        )
        # OrthoConv2d(
        #     in_channels=in_channels,
        #     out_channels=out_channels,
        #     kernel_size=3,
        #     stride=stride,
        #     padding=1,
        # )
    )
    # if act is not None:
    #     layers.append(act())
    if norm is not None:
        layers.append(norm(out_channels))
    for _ in range(n_blocks - 1):
        layers.append(
            # SDPBasedLipschitzResBlock(
            #     cin = out_channels,
            #     cout=out_channels,
            #     inner_dim_factor=0.25,
            #     kernel_size=3,
            #     stride=1,
            # )
            SLLxAOCLipschitzResBlock(
                cin=out_channels,
                inner_dim_factor=2.0,
                kernel_size=3,
            )
        )
        # if act is not None:
        #     layers.append(act())
        if norm is not None:
            layers.append(norm(out_channels))
    return layers


# def dumbNet500M(
#     img_size=(224, 224),
#     dim=1024,
#     depth=8,
#     kernel_size=5,
#     patch_size=16,
#     expand_factor=2,
#     n_classes=1000,
#     conv=ClassParam(
#         OrthoConv2d,
#         padding="same",
#         padding_mode="zeros",
#         bias=False,
#         bjorck_params=BjorckParams(
#             power_it_niter=3,
#             eps=1e-6,
#             bjorck_iters=6,
#             beta=0.5,
#         ),
#     ),
#     act=ClassParam(MaxMin),
#     lin=ClassParam(OrthoLinear, bias=False),
#     norm=ClassParam(BatchCentering2D),
#     pool=ClassParam(nn.LPPool2d, norm_type=2),
# ):
#     return nn.Sequential(
#         conv(
#             in_channels=3,
#             out_channels=dim,
#             kernel_size=patch_size,
#             stride=patch_size,
#             padding="valid",
#         ),
#         norm(num_features=dim) if norm is not None else nn.Identity(),
#         *[
#             Residual(
#                 nn.Sequential(
#                     conv(
#                         in_channels=dim,
#                         out_channels=expand_factor * dim,
#                         kernel_size=kernel_size,
#                     ),
#                     (
#                         norm(num_features=expand_factor * dim)
#                         if norm is not None
#                         else nn.Identity()
#                     ),
#                     act(),
#                     conv(
#                         in_channels=expand_factor * dim,
#                         kernel_size=kernel_size,
#                         out_channels=dim,
#                     ),
#                     norm(num_features=dim) if norm is not None else nn.Identity(),
#                 )
#             )
#             for i in range(depth)
#         ],
#         pool(kernel_size=(img_size[0] // patch_size, img_size[1] // patch_size)),
#         nn.Flatten(),
#         lin(
#             dim,
#             n_classes,
#         ),
#         # norm(num_features=n_classes) if norm is not None else nn.Identity(),
#     )


def AOCNetV1(
    img_shape=(3, 224, 224),
    n_classes=1000,
    expand_factor=2,
    block_depth=2,
    kernel_size=3,
    embedding_dim=1024,
    groups=8,
    skip=ClassParam(
        PrescaledAdditiveResidual,
        init_val=1.0,
    ),
    conv=ClassParam(
        AdaptiveOrthoConv2d,
        bias=False,
        padding="same",
        padding_mode="zeros",
        ortho_params=DEFAULT_ORTHO_PARAMS,
    ),
    act=ClassParam(MaxMin),
    lin=ClassParam(UnitNormLinear, bias=False),
    norm=ClassParam(BatchCentering2D),
    pool=ClassParam(nn.LPPool2d, norm_type=2),
):
    def resblock(in_channels, out_channels, n_blocks, conv, act, norm):
        layers = []
        if in_channels != out_channels:
            layers.append(
                conv(
                    in_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    stride=2,
                    padding=kernel_size // 2,
                )
            )
            # layers.append(act())
        layers.append(
            norm(num_features=out_channels) if norm is not None else nn.Identity()
        )
        for _ in range(n_blocks):
            res_layers = []
            res_layers.append(
                conv(
                    out_channels,
                    expand_factor * out_channels,
                    kernel_size=kernel_size,
                    groups=groups if groups is not None else out_channels // 2,
                )
            )
            res_layers.append(
                norm(num_features=expand_factor * out_channels)
                if norm is not None
                else nn.Identity()
            )
            res_layers.append(act())

            if groups is None or groups > 1 or expand_factor > 1:
                res_layers.append(
                    conv(
                        expand_factor * out_channels,
                        out_channels,
                        kernel_size=kernel_size,
                        groups=groups if groups is not None else out_channels // 2,
                    )
                )
            # if expand_factor > 1:
            #     res_layers.append(
            #         conv(
            #             expand_factor * out_channels,
            #             out_channels,
            #             kernel_size=kernel_size,
            #             groups=groups if groups is not None else out_channels // 2,
            #         )
            #     )
            if skip is not None:
                layers.append(skip(nn.Sequential(*res_layers)))
            else:
                layers.append(nn.Sequential(*res_layers))

            if groups is None or groups > 1:
                layers.append(conv(out_channels, out_channels, kernel_size=1))
                # layers.append(act())
                # layers.append(norm() if norm is not None else nn.Identity())
        return layers

    layers = [
        conv(
            in_channels=img_shape[0],
            out_channels=embedding_dim // 16,
            kernel_size=7,
            stride=2,
            padding=7 // 2,
        ),
        act(),
        norm(num_features=embedding_dim // 8) if norm is not None else nn.Identity(),
        conv(
            in_channels=embedding_dim // 16,
            out_channels=embedding_dim // 8,
            kernel_size=3,
            stride=2,
            padding=3 // 2,
        ),
        act(),
        norm(num_features=embedding_dim // 8) if norm is not None else nn.Identity(),
        # nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        *resblock(embedding_dim // 8, embedding_dim // 8, block_depth, conv, act, norm),
        *resblock(embedding_dim // 8, embedding_dim // 4, block_depth, conv, act, norm),
        *resblock(embedding_dim // 4, embedding_dim // 2, block_depth, conv, act, norm),
        *resblock(embedding_dim // 2, embedding_dim, block_depth, conv, act, norm),
        # nn.AvgPool2d(7, divisor_override=7),
        # nn.AdaptiveAvgPool2d((1, 1)),
        pool(kernel_size=(7, 7), stride=(7, 7)),
        nn.Flatten(),
        lin(embedding_dim, n_classes),
    ]
    return nn.Sequential(*layers)


SplitConcatNetConfigs = {
    "M": dict(
        expand_factor=4,
        block_depth=2,
        kernel_size=3,
        embedding_dim=1024,
        groups=32,
        conv=ClassParam(
            AdaptiveOrthoConv2d,
            bias=False,
            padding="same",
            padding_mode="zeros",
        ),
        skip=ClassParam(
            PrescaledAdditiveResidual,
            init_val=2.0,
        ),
        act=ClassParam(MaxMin),
        lin=ClassParam(UnitNormLinear, bias=False),
        norm=ClassParam(BatchCentering2D),
        pool=ClassParam(nn.LPPool2d, norm_type=2),
    ),
    "M2": dict(
        expand_factor=2,
        block_depth=2,
        kernel_size=5,
        embedding_dim=1024,
        groups=4,
        skip=ClassParam(
            PrescaledAdditiveResidual,
            init_val=2.0,
        ),
        conv=ClassParam(
            AdaptiveOrthoConv2d,
            bias=False,
            padding="same",
            padding_mode="zeros",
            ortho_params=DEFAULT_ORTHO_PARAMS,
        ),
        act=ClassParam(MaxMin),
        lin=ClassParam(UnitNormLinear, bias=False),
        norm=None,
        # norm=ClassParam(LayerCentering2D),
        pool=ClassParam(nn.AvgPool2d, divisor_override=7),
        # pool=ClassParam(nn.LPPool2d, norm_type=2),
    ),
    "M3": dict(
        expand_factor=2,
        block_depth=3,
        kernel_size=5,
        embedding_dim=2048,
        groups=None,  # None is depthwise, 1 is no groups
        # skip=None,
        skip=ClassParam(
            PrescaledAdditiveResidual,
            init_val=3.0,
        ),
        conv=ClassParam(
            AdaptiveOrthoConv2d,
            bias=False,
            padding="same",
            padding_mode="zeros",
            ortho_params=DEFAULT_ORTHO_PARAMS,
        ),
        act=ClassParam(MaxMin),
        lin=ClassParam(UnitNormLinear, bias=False),
        norm=None,
        # norm=ClassParam(LayerCentering2D),
        pool=ClassParam(nn.AvgPool2d, divisor_override=7),
        # pool=ClassParam(nn.LPPool2d, norm_type=2),
    ),
    "M4": dict(
        expand_factor=2,
        block_depth=3,
        kernel_size=5,
        embedding_dim=2048,
        groups=None,  # None is depthwise, 1 is no groups
        # skip=None,
        skip=ClassParam(
            PrescaledAdditiveResidual,
            init_val=3.0,
        ),
        conv=ClassParam(
            AdaptiveOrthoConv2d,
            bias=False,
            padding="same",
            padding_mode="circular",
            ortho_params=DEFAULT_ORTHO_PARAMS,
        ),
        act=ClassParam(MaxMin),
        lin=ClassParam(UnitNormLinear, bias=False),
        norm=None,
        # norm=ClassParam(LayerCentering2D),
        # pool=ClassParam(nn.AvgPool2d, divisor_override=7),
        pool=ClassParam(nn.LPPool2d, norm_type=2),
    ),
    "L": dict(
        expand_factor=2,
        block_depth=3,
        kernel_size=5,
        embedding_dim=3072,
        groups=None,  # None is depthwise, 1 is no groups
        # skip=None,
        skip=ClassParam(
            PrescaledAdditiveResidual,
            init_val=3.0,
        ),
        conv=ClassParam(
            AdaptiveOrthoConv2d,
            bias=False,
            padding="same",
            padding_mode="zeros",
            ortho_params=DEFAULT_ORTHO_PARAMS,
        ),
        act=ClassParam(MaxMin),
        lin=ClassParam(UnitNormLinear, bias=False),
        norm=None,
        # norm=ClassParam(LayerCentering2D),
        pool=ClassParam(nn.AvgPool2d, divisor_override=7),
        # pool=ClassParam(nn.LPPool2d, norm_type=2),
    ),
}


def LipResNet(
    img_shape=(3, 224, 224),
    n_classes=1000,
    skip=ClassParam(
        PrescaledAdditiveResidual,
        init_val=3.0,
    ),
    conv=ClassParam(
        AdaptiveOrthoConv2d,
        bias=False,
        padding="same",
        padding_mode="zeros",
        ortho_params=DEFAULT_ORTHO_PARAMS,
    ),
    act=ClassParam(MaxMin),
    lin=ClassParam(UnitNormLinear, bias=False),
    norm=None,  # ClassParam(BatchCentering2D),
    # pool=ClassParam(nn.LPPool2d, norm_type=2),
):
    layers = [
        conv(
            in_channels=img_shape[0],
            out_channels=64,
            kernel_size=7,
            stride=2,
            padding=3,
        ),
        act(),
        norm() if norm is not None else nn.Identity(),
        # nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
        *ResNetBlock(64, 64, 3, conv, skip, act, norm),
        *ResNetBlock(64, 128, 4, conv, skip, act, norm),
        *ResNetBlock(128, 256, 6, conv, skip, act, norm),
        *ResNetBlock(256, 512, 3, conv, skip, act, norm),
        nn.AvgPool2d(7, divisor_override=7),
        # nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),
        lin(512, n_classes),
    ]
    return nn.Sequential(*layers)


def ResNetBlock(in_channels, out_channels, n_blocks, stridedconvn, skipconv, act, norm):
    layers = []
    if in_channels != out_channels:
        layers.append(
            stridedconvn(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
        )
        n_blocks -= 1
        # layers.append(act())
        layers.append(norm() if norm is not None else nn.Identity())
    for _ in range(n_blocks):
        layers.append(
            skipconv(
                nn.Sequential(
                    *[
                        stridedconvn(
                            out_channels, out_channels, kernel_size=3, padding=1
                        ),
                        norm() if norm is not None else nn.Identity(),
                        act(),
                        stridedconvn(
                            out_channels, out_channels, kernel_size=3, padding=1
                        ),
                        norm() if norm is not None else nn.Identity(),
                        act(),
                    ]
                )
            )
        )
    return layers


def LipVGG(
    img_shape=(3, 224, 224),
    n_classes=1000,
    conv=ClassParam(
        AdaptiveOrthoConv2d,
        bias=False,
        padding="same",
        padding_mode="zeros",
    ),
    act=ClassParam(MaxMin),
    lin=ClassParam(OrthoLinear, bias=False),
    norm=ClassParam(LayerCentering2D),
):
    layers = [
        conv(in_channels=img_shape[0], out_channels=64, kernel_size=3),
        act(),
        norm() if norm is not None else nn.Identity(),
        conv(in_channels=64, out_channels=128, kernel_size=3, stride=2),
        act(),
        norm() if norm is not None else nn.Identity(),
        conv(in_channels=128, out_channels=128, kernel_size=3),
        act(),
        norm() if norm is not None else nn.Identity(),
        conv(in_channels=128, out_channels=256, kernel_size=3, stride=2),
        act(),
        norm() if norm is not None else nn.Identity(),
        conv(in_channels=256, out_channels=256, kernel_size=3),
        act(),
        norm() if norm is not None else nn.Identity(),
        conv(in_channels=256, out_channels=256, kernel_size=3),
        act(),
        norm() if norm is not None else nn.Identity(),
        conv(in_channels=256, out_channels=512, kernel_size=3, stride=2),
        act(),
        norm() if norm is not None else nn.Identity(),
        conv(in_channels=512, out_channels=512, kernel_size=3),
        act(),
        norm() if norm is not None else nn.Identity(),
        conv(in_channels=512, out_channels=512, kernel_size=3),
        act(),
        norm() if norm is not None else nn.Identity(),
        conv(in_channels=512, out_channels=512, kernel_size=3, stride=2),
        act(),
        norm() if norm is not None else nn.Identity(),
        conv(in_channels=512, out_channels=512, kernel_size=3),
        act(),
        norm() if norm is not None else nn.Identity(),
        conv(in_channels=512, out_channels=512, kernel_size=3),
        act(),
        norm() if norm is not None else nn.Identity(),
        conv(in_channels=512, out_channels=512, kernel_size=3, stride=2),
        nn.Flatten(),  # fm = 7x7
        lin(512 * 7 * 7, 4096),
        act(),
        norm(dim=-1) if norm is not None else nn.Identity(),
        lin(4096, 4096),
        act(),
        norm(dim=-1) if norm is not None else nn.Identity(),
        lin(4096, n_classes),
    ]
    return nn.Sequential(*layers)


def PatchBasedCNN(
    img_shape=(3, 32, 32),
    dim=128,
    depth=8,
    kernel_size=3,
    patch_size=2,
    n_classes=10,
    groups=1,
    conv=ClassParam(
        AdaptiveOrthoConv2d,
        bias=False,
        padding="same",
    ),
    act=ClassParam(MaxMin),
    pool=ClassParam(nn.LPPool2d, norm_type=2),
    lin=ClassParam(OrthoLinear),
    norm=ClassParam(LayerCentering2D),
):
    return nn.Sequential(
        conv(
            in_channels=img_shape[0],
            out_channels=dim,
            kernel_size=patch_size,
            stride=patch_size,
            padding="valid",
            padding_mode="zeros",
        ),
        act(),
        norm() if norm is not None else nn.Identity(),
        *[
            nn.Sequential(
                conv(
                    in_channels=dim,
                    out_channels=dim,
                    kernel_size=kernel_size,
                    groups=groups,
                ),
                norm() if norm is not None else nn.Identity(),
                act(),
                # (
                #     ChannelShuffle(g, dim // g)
                #     if (g := (groups if i % 2 == 0 else dim // groups))
                #     > 1  # number of group switch every layer
                #     else nn.Identity()
                # ),
            )
            for i in range(depth)
        ],
        # AvgPool2d is AvgPool2d but with a sqrt(w*h)
        # factor, as it would be 1/sqrt(w,h) lip otherwise
        pool(
            kernel_size=(img_shape[1] // patch_size, img_shape[2] // patch_size),
            stride=None,
        ),
        nn.Flatten(),
        lin(
            dim,
            n_classes,
        ),
    )


def PatchBasedExapandedCNN(
    img_shape=(3, 224, 224),
    dim=128,
    depth=12,
    kernel_size=5,
    patch_size=2,
    expand_factor=2,
    groups=None,
    n_classes=1000,
    skip=False,
    conv=ClassParam(
        AdaptiveOrthoConv2d,
        bias=False,
        padding="same",
    ),
    act=ClassParam(MaxMin),
    pool=ClassParam(AvgPool2d),
    lin=ClassParam(OrthoLinear),
    norm=ClassParam(LayerCentering2D),
):
    if skip:
        skipco = PrescaledAdditiveResidual
    else:
        skipco = nn.Sequential
    return nn.Sequential(
        conv(
            in_channels=img_shape[0],
            out_channels=dim,
            kernel_size=patch_size,
            stride=patch_size,
            padding="valid",
        ),
        *[
            skipco(
                nn.Sequential(
                    conv(
                        in_channels=dim,
                        out_channels=int(dim * expand_factor),
                        kernel_size=kernel_size,
                        groups=(
                            groups
                            if groups is not None
                            else min(int(dim * expand_factor) // 2, dim // 2)
                        ),
                    ),
                    act(),
                    norm() if norm is not None else nn.Identity(),
                    # (
                    #     ChannelShuffle(groups, dim * expand_factor // groups)
                    #     if groups > 1
                    #     else nn.Identity()
                    # ),
                    conv(
                        in_channels=int(dim * expand_factor),
                        out_channels=dim,
                        kernel_size=1,
                        groups=1,
                    ),
                    # ChannelShuffle(dim // groups, groups) if groups > 1 else nn.Identity(),
                )
            )
            for i in range(depth)
        ],
        # AvgPool2d is AvgPool2d but with a sqrt(w*h)
        # factor, as it would be 1/sqrt(w,h) lip otherwise
        pool(),  # ((img_shape[1] // patch_size, img_shape[2] // patch_size), None),
        nn.Flatten(),
        act(),
        lin(
            dim,
            n_classes,
        ),
    )


def ConvMixerInspired(
    img_shape=(3, 32, 32),
    dim=128,
    depth=8,
    kernel_size=3,
    patch_size=2,
    expand_factor=1,
    channels_per_group=1,
    n_classes=10,
    conv=ClassParam(
        AdaptiveOrthoConv2d,
        bias=False,
        padding="same",
    ),
    act=ClassParam(MaxMin),
    pool=ClassParam(AvgPool2d),
    lin=ClassParam(OrthoLinear),
    norm=ClassParam(LayerCentering2D),
):
    return nn.Sequential(
        conv(
            in_channels=img_shape[0],
            out_channels=dim,
            kernel_size=patch_size,
            stride=patch_size,
            padding="valid",
        ),
        *[
            nn.Sequential(
                # depthwise (with groups of channels_per_group channels)
                conv(
                    in_channels=dim,
                    out_channels=dim * expand_factor,
                    groups=(dim * expand_factor) // channels_per_group,
                    kernel_size=kernel_size,
                ),
                act(),
                norm() if norm is not None else nn.Identity(),
                # pointwise
                conv(
                    kernel_size=1,
                    in_channels=dim * expand_factor,
                    out_channels=dim,
                ),
                act(),
                norm() if norm is not None else nn.Identity(),
            )
            for i in range(depth)
        ],
        # AvgPool2d is AvgPool2d but with a sqrt(w*h)
        # factor, as it would be 1/sqrt(w,h) lip otherwise
        pool((img_shape[1] // patch_size, img_shape[2] // patch_size), None),
        nn.Flatten(),
        lin(
            dim,
            n_classes,
        ),
    )


def BCOPLargeCNN(
    img_shape=(3, 32, 32),
    n_classes=10,
    conv=ClassParam(
        AdaptiveOrthoConv2d,
        bias=True,
        padding="circular",
    ),
    act=ClassParam(MaxMin),
    lin=ClassParam(OrthoLinear),
    norm=None,
):
    layers = [
        conv(
            in_channels=img_shape[0],
            out_channels=32,
            kernel_size=3,
            stride=1,
            padding=1,
        ),
        act(),
        norm() if norm is not None else nn.Identity(),
        conv(in_channels=32, out_channels=64, kernel_size=5, stride=2, padding=2),
        act(),
        norm() if norm is not None else nn.Identity(),
        conv(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
        act(),
        norm() if norm is not None else nn.Identity(),
        conv(in_channels=64, out_channels=64, kernel_size=5, stride=2, padding=2),
        act(),
        norm() if norm is not None else nn.Identity(),
        nn.Flatten(),
        lin((img_shape[1] // 4) * (img_shape[2] // 4) * 64, 512),
        act(),
        lin(512, 512),
        act(),
        lin(512, n_classes),
    ]

    return nn.Sequential(*layers)


def StagedCNN(
    img_shape=(3, 32, 32),
    dim_repeats=[(64, 2), (128, 2)],
    dim_nb_dense=(256, 2),
    n_classes=10,
    conv=ClassParam(
        AdaptiveOrthoConv2d,
        bias=False,
        padding="same",
    ),
    act=ClassParam(MaxMin),
    pool=None,
    lin=ClassParam(OrthoLinear),
    norm=ClassParam(LayerCentering2D),
):
    layers = []
    in_channels = img_shape[0]

    useSharedLip = False
    if isinstance(norm, ClassParam) and issubclass(norm.fct,ol.ScaledLipschitzModule):
        scaleLipFactory = ol.SharedLipFactory()
        norm.kwargs["factory"] = scaleLipFactory
        useSharedLip = True

    # enrich the list of dim_repeats with the next number of channels in the list
    for i in range(len(dim_repeats) - 1):
        dim_repeats[i] = (dim_repeats[i][0], dim_repeats[i][1], dim_repeats[i + 1][0])
    dim_repeats[-1] = (dim_repeats[-1][0], dim_repeats[-1][1], None)

    # Create convolutional blocks
    for dim, repeats, next_dim in dim_repeats:
        # Add repeated conv layers
        for _ in range(repeats):
            layers.append(conv(in_channels=in_channels, out_channels=dim))
            layers.append(norm(num_features = dim) if norm is not None else nn.Identity())
            layers.append(act())
            in_channels = dim

        if next_dim is not None:
            # Add strided convolution to separate blocks
            layers.append(
                conv(
                    in_channels=dim,
                    out_channels=next_dim,
                    stride=2,
                )
            )
            layers.append(norm(num_features = next_dim) if norm is not None else nn.Identity())
            layers.append(act())
            in_channels = next_dim

    feat_shape = img_shape[-1] // (2 ** (len(dim_repeats) - 1))
    if pool is not None:
        layers.append(pool(kernel_size=feat_shape))
        feat_shape = 1
    # Flatten layer
    layers.append(nn.Flatten())
    nb_features = dim * feat_shape**2
    if dim_nb_dense is not None and len(dim_nb_dense) > 0:
        # Add linear layers
        dim, repeats = dim_nb_dense
        for _ in range(repeats):
            layers.append(lin(nb_features, dim))
            layers.append(norm(num_features = dim) if norm is not None else nn.Identity())
            layers.append(act())
            nb_features = dim
    else:
        dim = nb_features
    # Final linear layer for classification
    layers.append(lin(dim, n_classes))

    if useSharedLip:
        print(layers)
        return ol.BnLipSequential(lipFactory=scaleLipFactory, layers=layers)
    else:
        return nn.Sequential(*layers)


def LargeStagedCNN(
    img_shape=(3, 224, 224),
    patch_size=7,
    dim_repeats=[(128, 4), (256, 4), (512, 4), (1024, 4)],
    dim_nb_dense=(1024, 5),
    n_classes=1000,
    conv=ClassParam(
        AdaptiveOrthoConv2d,
        bias=False,
        padding_mode="circular",
        kernel_size=3,
        padding=1,
    ),
    act=ClassParam(MaxMin),
    lin=ClassParam(UnitNormLinear, bias=False),
    norm=None,
    pool=None,
):
    layers = []

    layers.append(
        conv(
            in_channels=img_shape[0],
            out_channels=dim_repeats[0][0],
            kernel_size=patch_size,
            stride=patch_size,
            padding=0,
        )
    )
    layers.append(act())
    layers.append(norm() if norm is not None else nn.Identity())

    in_channels = dim_repeats[0][0]

    # enrich the list of dim_repeats with the next number of channels in the list
    for i in range(len(dim_repeats) - 1):
        dim_repeats[i] = (dim_repeats[i][0], dim_repeats[i][1], dim_repeats[i + 1][0])
    dim_repeats[-1] = (dim_repeats[-1][0], dim_repeats[-1][1], None)

    # Create convolutional blocks
    for dim, repeats, next_dim in dim_repeats:
        # Add repeated conv layers
        for _ in range(repeats):
            layers.append(conv(in_channels=in_channels, out_channels=dim))
            layers.append(act())
            layers.append(norm() if norm is not None else nn.Identity())
            in_channels = dim

        if next_dim is not None:
            # Add strided convolution to separate blocks
            layers.append(
                conv(
                    in_channels=dim,
                    out_channels=next_dim,
                    stride=2,
                )
            )
            layers.append(act())
            layers.append(norm() if norm is not None else nn.Identity())
            in_channels = next_dim

    feat_shape = img_shape[-1] // (patch_size * (2 ** (len(dim_repeats) - 1)))
    if pool is not None:
        layers.append(pool(kernel_size=feat_shape))
        feat_shape = 1
    # Flatten layer
    layers.append(nn.Flatten())
    nb_features = dim * feat_shape**2
    if dim_nb_dense is not None and len(dim_nb_dense) > 0:
        # Add linear layers
        dim, repeats = dim_nb_dense
        for _ in range(repeats):
            layers.append(lin(nb_features, dim))
            layers.append(act())
            nb_features = dim
    else:
        dim = nb_features
    # Final linear layer for classification
    layers.append(lin(dim, n_classes))

    return nn.Sequential(*layers)


MODELS = {
    "SplitConcatNet-M": lambda *args, **kwargs: AOCNetV1(
        *args, **kwargs, **SplitConcatNetConfigs["M"]
    ),
    "SplitConcatNet-M2": lambda *args, **kwargs: AOCNetV1(
        *args, **kwargs, **SplitConcatNetConfigs["M2"]
    ),
    "LipResNet": LipResNet,
}
