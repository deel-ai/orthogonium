import math
import os

from lightning.pytorch import callbacks as pl_callbacks
from lightning.pytorch import Trainer
from lightning.pytorch import LightningModule
from lightning.pytorch import LightningDataModule
import schedulefree
import torch.utils.data
import torchmetrics
from lightning.pytorch.loggers import WandbLogger
from torch.utils.data import DataLoader
from torchinfo import summary
from torchvision.datasets import ImageFolder
from torchvision.transforms import CenterCrop
from torchvision.transforms import Compose
from torchvision.transforms import Normalize
from torchvision.transforms import RandomHorizontalFlip
from torchvision.transforms import RandomResizedCrop
from torchvision.transforms import Resize
from torchvision.transforms import ToTensor

from orthogonium.model_factory.classparam import ClassParam
from orthogonium.layers import UnitNormLinear
from orthogonium.layers.conv.AOC import AdaptiveOrthoConv2d
from orthogonium.layers.custom_activations import MaxMin
from orthogonium.reparametrizers import DEFAULT_ORTHO_PARAMS, QR_ORTHO_PARAMS
from orthogonium.losses import check_last_linear_layer_type
from orthogonium.losses import LossXent
from orthogonium.losses import VRA
from orthogonium.model_factory.models_factory import AOCNetV1
from orthogonium.layers.residual import PrescaledAdditiveResidual

torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("medium")


this_directory = os.path.abspath(os.path.dirname(__file__))
parent_directory = os.path.abspath(os.path.join(this_directory, os.pardir))

MAX_EPOCHS = 300  # as done in resnet strikes back


class ImagenetDataModule(LightningDataModule):
    # Dataset configuration
    _DATA_PATH = f"/local_data/imagenet_cache/ILSVRC/Data/CLS-LOC/"
    _BATCH_SIZE = 256
    _NUM_WORKERS = 8  # Number of parallel processes fetching data
    _PREPROCESSING_PARAMS = {
        # "img_mean": (0.41757566, 0.26098573, 0.25888634),
        # "img_std": (0.21938758, 0.1983, 0.19342837),
        "img_mean": (0.5, 0.5, 0.5),
        "img_std": (0.5, 0.5, 0.5),
        "crop_size": 224,
        "horizontal_flip_prob": 0.5,
        # "randaug_params": {"magnitude": 8, "num_ops": 2},
        "random_resized_crop_params": {
            "scale": (0.5, 1.0),
            "ratio": (3.0 / 4.0, 4.0 / 3.0),
        },
    }

    def train_dataloader(self):
        # Define the transformations
        transform = Compose(
            [
                Resize(256),
                RandomResizedCrop(
                    self._PREPROCESSING_PARAMS["crop_size"],
                    **self._PREPROCESSING_PARAMS["random_resized_crop_params"],
                ),
                RandomHorizontalFlip(
                    self._PREPROCESSING_PARAMS["horizontal_flip_prob"]
                ),
                # RandAugment(**self._PREPROCESSING_PARAMS["randaug_params"]),
                ToTensor(),
                Normalize(
                    mean=self._PREPROCESSING_PARAMS["img_mean"],
                    std=self._PREPROCESSING_PARAMS["img_std"],
                ),
            ]
        )

        # Load the dataset
        train_dataset = ImageFolder(
            self._DATA_PATH + "train",
            transform=transform,
        )

        return DataLoader(
            train_dataset,
            batch_size=self._BATCH_SIZE,
            num_workers=self._NUM_WORKERS,
            prefetch_factor=2,
            shuffle=True,
        )

    def val_dataloader(self):
        # Define the transformations
        transform = Compose(
            [
                Resize(256),
                CenterCrop(self._PREPROCESSING_PARAMS["crop_size"]),
                ToTensor(),
                Normalize(
                    mean=self._PREPROCESSING_PARAMS["img_mean"],
                    std=self._PREPROCESSING_PARAMS["img_std"],
                ),
            ]
        )

        # Load the dataset
        val_dataset = ImageFolder(
            self._DATA_PATH + "val",
            transform=transform,
        )

        return DataLoader(
            val_dataset,
            batch_size=self._BATCH_SIZE,
            num_workers=self._NUM_WORKERS,
            shuffle=False,
        )


class ClassificationLightningModule(LightningModule):
    def __init__(self, num_classes=1000):
        super().__init__()
        self.num_classes = num_classes
        self.model = AOCNetV1(
            img_shape=(3, 224, 224),
            n_classes=num_classes,
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
                ortho_params=QR_ORTHO_PARAMS,
            ),
            act=ClassParam(MaxMin),
            lin=ClassParam(UnitNormLinear, bias=False),
            norm=None,
            pool=None,  # ClassParam(nn.LPPool2d, norm_type=2),
        )
        # self.criteria = CosineLoss()
        # self.criteria = LossXent(num_classes, offset=0, temperature=0.5 * 0.125)
        self.criteria = LossXent(
            num_classes, offset=1.5 * math.sqrt(2), temperature=0.25
        )
        self.train_acc = torchmetrics.Accuracy(
            task="multiclass", num_classes=num_classes
        )
        self.val_acc = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        self.train_vra = torchmetrics.MeanMetric()
        self.val_vra = torchmetrics.MeanMetric()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        self.model.train()
        if hasattr(self.optimizers(), "train"):
            self.optimizers().train()
        img, label = batch
        y_hat = self.model(img)
        loss = self.criteria(y_hat, label)
        self.train_acc(y_hat, label)
        self.train_vra(
            VRA(
                y_hat,
                label,
                L=1 / min(ImagenetDataModule._PREPROCESSING_PARAMS["img_std"]),
                eps=36 / 255,
                last_layer_type=check_last_linear_layer_type(self.model),
            )
        )  # L is 1 / min std of imagenet
        self.log(
            "loss",
            loss,
            on_epoch=True,
            on_step=True,
            prog_bar=True,
            sync_dist=True,
        )
        self.log(
            "accuracy",
            self.train_acc,
            on_epoch=True,
            on_step=True,
            prog_bar=True,
            sync_dist=True,
        )
        self.log(
            "vra",
            self.train_vra,
            on_epoch=True,
            on_step=True,
            prog_bar=True,
            sync_dist=False,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        self.model.eval()
        if hasattr(self.optimizers(), "eval"):
            self.optimizers().eval()
        img, label = batch
        y_hat = self.model(img)
        loss = self.criteria(y_hat, label)
        self.val_acc(y_hat, label)
        self.val_vra(
            VRA(
                y_hat,
                label,
                L=1 / min(ImagenetDataModule._PREPROCESSING_PARAMS["img_std"]),
                eps=36 / 255,
                last_layer_type=check_last_linear_layer_type(self.model),
            )
        )  # L is 1 / min std of imagenet
        self.log(
            "val_loss",
            loss,
            on_epoch=True,
            on_step=False,
            prog_bar=True,
            sync_dist=True,
        )
        self.log(
            "val_accuracy",
            self.val_acc,
            on_epoch=True,
            on_step=False,
            prog_bar=True,
            sync_dist=True,
        )
        self.log(
            "val_vra",
            self.val_vra,
            on_epoch=True,
            on_step=False,
            prog_bar=True,
            sync_dist=True,
        )
        return loss

    def on_fit_start(self) -> None:
        if hasattr(self.optimizers(), "train"):
            self.optimizers().train()

    def on_predict_start(self) -> None:
        if hasattr(self.optimizers(), "eval"):
            self.optimizers().eval()

    def on_validation_model_eval(self) -> None:
        self.model.eval()
        if hasattr(self.optimizers(), "eval"):
            self.optimizers().eval()

    def on_validation_model_train(self) -> None:
        self.model.train()
        if hasattr(self.optimizers(), "train"):
            self.optimizers().train()

    def on_test_model_eval(self) -> None:
        self.model.eval()
        if hasattr(self.optimizers(), "eval"):
            self.optimizers().eval()

    def on_test_model_train(self) -> None:
        self.model.train()
        if hasattr(self.optimizers(), "train"):
            self.optimizers().train()

    def on_predict_model_eval(self) -> None:  # redundant with on_predict_start()
        self.model.eval()
        if hasattr(self.optimizers(), "eval"):
            self.optimizers().eval()

    def configure_optimizers(self):
        """
        Setup the Adam optimizer. Note, that this function also can return a lr scheduler, which is
        usually useful for training video models.
        """
        # return torch.optim.AdamW(self.parameters(), lr=0.0001, weight_decay=1e-5)
        optimizer = schedulefree.AdamWScheduleFree(
            self.parameters(), lr=5e-3, weight_decay=0
        )
        optimizer.train()
        self.hparams["lr"] = optimizer.param_groups[0]["lr"]
        return optimizer


def train():
    classification_module = ClassificationLightningModule(num_classes=1000)
    data_module = ImagenetDataModule()
    wandb_logger = WandbLogger(project="lipschitz-robust-imagenet", log_model=True)
    # wandb_logger.experiment.config["ortho_method"] = method
    # wandb_logger.experiment.config["criteria"] = criteria
    checkpoint_callback = pl_callbacks.ModelCheckpoint(
        monitor="loss",
        mode="min",
        save_top_k=1,
        save_last=True,
        dirpath=f"./checkpoints/{wandb_logger.experiment.dir}",
    )
    trainer = Trainer(
        accelerator="gpu",
        devices=-1,  # GPUs per node
        num_nodes=1,  # Number of nodes
        # num_nodes=3,  # Number of nodes
        strategy="ddp",  # Distributed strategy
        precision="bf16-mixed",
        max_epochs=MAX_EPOCHS,
        enable_model_summary=True,
        logger=[wandb_logger],
        # logger=False,
        callbacks=[
            # pl_callbacks.LearningRateFinder(max_lr=0.05),
            checkpoint_callback,
        ],
    )
    summary(classification_module, input_size=(1, 3, 224, 224))

    trainer.fit(classification_module, data_module)
    # save the model
    # torch.save(classification_module.model.state_dict(), "single_stage.pth")


if __name__ == "__main__":
    train()
