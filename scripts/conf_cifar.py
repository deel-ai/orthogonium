import math
import copy

from orthogonium.model_factory.classparam import ClassParam
from orthogonium.layers.conv.AOC import AdaptiveOrthoConv2d
from orthogonium.layers.linear import OrthoLinear, UnitNormLinear
from orthogonium.layers.custom_activations import MaxMin, Abs
from orthogonium.layers.normalization import BatchLipNorm
from orthogonium.layers import BatchCentering
from orthogonium.losses import LossXent
from orthogonium.losses import VRA
from orthogonium.model_factory.models_factory import (
    StagedCNN,
)

from deel.torchlip import SoftHKRMulticlassLoss

stagecnn_lipbn = ClassParam(
    StagedCNN,
    img_shape=(3, 32, 32),
    dim_repeats=[(128, 4), (256, 4), (512, 4), (1024, 4)],
    dim_nb_dense=(1024, 5),
    n_classes=10,
    conv=ClassParam(
        AdaptiveOrthoConv2d,
        bias=False,
        padding_mode="circular",
        kernel_size=3,
        padding=1,
    ),
    act=ClassParam(MaxMin),
    lin=ClassParam(OrthoLinear, bias=True),
    norm=ClassParam(BatchLipNorm),
)

stagecnn_lipbn_disjoint = copy.deepcopy(stagecnn_lipbn)
stagecnn_lipbn_disjoint.kwargs["lin"] = ClassParam(UnitNormLinear, bias=True)


stagecnn_center = copy.deepcopy(stagecnn_lipbn)
stagecnn_center.kwargs["norm"] = ClassParam(BatchCentering)


stagecnn_center_disjoint = copy.deepcopy(stagecnn_center)
stagecnn_center_disjoint.kwargs["lin"] = ClassParam(UnitNormLinear, bias=True)

loss_xent_robust = ClassParam(
    LossXent, n_classes=10, offset=1.5 * math.sqrt(2), temperature=0.25
)


loss_hkr = ClassParam(
    SoftHKRMulticlassLoss,
    alpha=0.9995,
    min_margin=1.2,  # 1.0, #36 / 255.0,
    temperature=0.25,
)


conf_stagedcnn_lipbn_robust = {
    "conf_name": "stagedcnn_lipbn_robust",
    "model": stagecnn_lipbn,
    "loss": loss_xent_robust,
    "last_layer_type": "global",
    "epochs": 200,
}

conf_stagedcnn_lipbn_robust_hkr = copy.deepcopy(conf_stagedcnn_lipbn_robust)
conf_stagedcnn_lipbn_robust_hkr["conf_name"] = "stagedcnn_lipbn_robust_hkr"
conf_stagedcnn_lipbn_robust_hkr["loss"] = loss_hkr
conf_stagedcnn_lipbn_robust_hkr["require_one_hot_labels"] = True


conf_stagedcnn_lipbn_disjoint_robust = {
    "conf_name": "stagedcnn_lipbn_disjoint_robust",
    "model": stagecnn_lipbn_disjoint,
    "loss": loss_xent_robust,
    "last_layer_type": "classwise",
    "epochs": 200,  # 200,
}

conf_stagedcnn_lipbn_disjoint_robust_hkr = copy.deepcopy(
    conf_stagedcnn_lipbn_disjoint_robust
)
conf_stagedcnn_lipbn_disjoint_robust_hkr["conf_name"] = (
    "stagedcnn_lipbn_disjoint_robust_hkr"
)
conf_stagedcnn_lipbn_disjoint_robust_hkr["loss"] = loss_hkr
conf_stagedcnn_lipbn_disjoint_robust_hkr["require_one_hot_labels"] = True

conf_stagedcnn_center_robust = {
    "conf_name": "stagedcnn_center_robust",
    "model": stagecnn_center,
    "loss": loss_xent_robust,
    "last_layer_type": "global",
    "epochs": 200,
}

conf_stagedcnn_center_robust_hkr = copy.deepcopy(conf_stagedcnn_center_robust)
conf_stagedcnn_center_robust_hkr["conf_name"] = "stagedcnn_center_robust_hkr"
conf_stagedcnn_center_robust_hkr["loss"] = loss_hkr
conf_stagedcnn_center_robust_hkr["require_one_hot_labels"] = True

conf_stagedcnn_center_disjoint_robust = {
    "conf_name": "stagedcnn_center_disjoint_robust",
    "model": stagecnn_center_disjoint,
    "loss": loss_xent_robust,
    "last_layer_type": "classwise",
    "epochs": 200,
}

conf_stagedcnn_center_disjoint_robust_hkr = copy.deepcopy(
    conf_stagedcnn_center_disjoint_robust
)
conf_stagedcnn_center_disjoint_robust_hkr["conf_name"] = (
    "stagedcnn_center_disjoint_robust_hkr"
)
conf_stagedcnn_center_disjoint_robust_hkr["loss"] = loss_hkr
conf_stagedcnn_center_disjoint_robust_hkr["require_one_hot_labels"] = True
