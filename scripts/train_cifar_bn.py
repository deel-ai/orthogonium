import math
import os
import sys
import inspect

# sys.path.append("./")
# sys.path.append("../deel-torchlip")

try:
    import wandb
except ImportError:
    wandb = None

import argparse
import schedulefree
import torch.utils.data
import torchmetrics
from lightning.pytorch import callbacks as pl_callbacks
from lightning.pytorch import Trainer
from lightning.pytorch import LightningModule
from lightning.pytorch import LightningDataModule
from lightning.pytorch.loggers import WandbLogger
from torch.utils.data import DataLoader
from torchinfo import summary
from torchvision.datasets import CIFAR10
from torchvision.transforms import Compose
from torchvision.transforms import Normalize
from torchvision.transforms import RandAugment
from torchvision.transforms import RandomHorizontalFlip
from torchvision.transforms import RandomResizedCrop
from torchvision.transforms import ToTensor
from lightning.pytorch.callbacks import LearningRateMonitor

from orthogonium.model_factory.classparam import ClassParam
from orthogonium.layers.conv.AOC import AdaptiveOrthoConv2d
from orthogonium.layers.linear import OrthoLinear
from orthogonium.layers.custom_activations import MaxMin, Abs
from orthogonium.layers.normalization import BatchLipNorm
from orthogonium.losses import LossXent
from orthogonium.losses import VRA
from orthogonium.model_factory.models_factory import (
    StagedCNN,
)

import conf_cifar

torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("medium")
from torch.optim import Adam

this_directory = os.path.abspath(os.path.dirname(__file__))
parent_directory = os.path.abspath(os.path.join(this_directory, os.pardir))

MAX_EPOCHS = 200  # 300 #3000  # might seem large, but this amounts to only 150k steps


def get_confs():
    confs = {}
    for mname, module in inspect.getmembers(conf_cifar):
        if isinstance(module, dict) and ("conf_name" in module.keys()):
            confs[module["conf_name"]] = module
    return confs


class Cifar10DataModule(LightningDataModule):
    # Dataset configuration
    _BATCH_SIZE = 256
    _NUM_WORKERS = 8  # Number of parallel processes fetching data
    _PREPROCESSING_PARAMS = {
        # "img_mean": (0.41757566, 0.26098573, 0.25888634),
        # "img_std": (0.21938758, 0.1983, 0.19342837),
        "img_mean": (0.5, 0.5, 0.5),
        "img_std": (0.5, 0.5, 0.5),
        "crop_size": 32,
        "horizontal_flip_prob": 0.5,
        "randaug_params": {"magnitude": 6, "num_ops": 1},
        "random_resized_crop_params": {
            "scale": (0.5, 1.0),
            "ratio": (3.0 / 4.0, 4.0 / 3.0),
        },
    }

    def train_dataloader(self):
        # Define the transformations
        transform = Compose(
            [
                RandomResizedCrop(
                    self._PREPROCESSING_PARAMS["crop_size"],
                    **self._PREPROCESSING_PARAMS["random_resized_crop_params"],
                ),
                RandomHorizontalFlip(
                    self._PREPROCESSING_PARAMS["horizontal_flip_prob"]
                ),
                RandAugment(**self._PREPROCESSING_PARAMS["randaug_params"]),
                ToTensor(),
                Normalize(
                    mean=self._PREPROCESSING_PARAMS["img_mean"],
                    std=self._PREPROCESSING_PARAMS["img_std"],
                ),
            ]
        )

        # Load the dataset
        train_dataset = CIFAR10(
            root="/datasets/pytorch_datasets/cifar10/",
            train=True,
            download=True,
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
                # Resize(256),
                # CenterCrop(self._PREPROCESSING_PARAMS["crop_size"]),
                ToTensor(),
                Normalize(
                    mean=self._PREPROCESSING_PARAMS["img_mean"],
                    std=self._PREPROCESSING_PARAMS["img_std"],
                ),
            ]
        )

        # Load the dataset
        val_dataset = CIFAR10(
            root="/datasets/pytorch_datasets/cifar10/",
            train=False,
            download=True,
            transform=transform,
        )

        return DataLoader(
            val_dataset,
            batch_size=self._BATCH_SIZE,
            num_workers=self._NUM_WORKERS,
            shuffle=False,
        )


class ClassificationLightningModule(LightningModule):
    def __init__(self, num_classes=10, classif_args=None):
        super().__init__()
        self.num_classes = num_classes
        self.model = classif_args["model"]()

        self.criteria = classif_args["loss"]()
        self.train_acc = torchmetrics.Accuracy(
            task="multiclass", num_classes=num_classes
        )
        self.val_acc = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        self.train_vra = torchmetrics.MeanMetric()
        self.val_vra = torchmetrics.MeanMetric()
        self.last_layer_type = classif_args["last_layer_type"]  # global
        self.require_one_hot_labels = False
        if "require_one_hot_labels" in classif_args:
            self.require_one_hot_labels = classif_args["require_one_hot_labels"]
        self.max_epochs = classif_args["epochs"]

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        self.model.train()
        if hasattr(self.optimizers(), "train"):
            self.optimizers().train()
        img, label = batch
        y_hat = self.model(img)
        if self.require_one_hot_labels:
            # one_hot encodeing of labels
            label_onehot = torch.nn.functional.one_hot(
                label, num_classes=self.num_classes
            ).float()
            loss = self.criteria(y_hat, label_onehot)
        else:
            loss = self.criteria(y_hat, label)
        self.train_acc(y_hat, label)
        self.train_vra(
            VRA(
                y_hat,
                label,
                L=1 / min(Cifar10DataModule._PREPROCESSING_PARAMS["img_std"]),
                eps=36 / 255,
                last_layer_type=self.last_layer_type,
            )
        )  # L is 1 / max std of imagenet
        # Log the train loss to Tensorboard

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
        if self.require_one_hot_labels:
            label_onehot = torch.nn.functional.one_hot(
                label, num_classes=self.num_classes
            ).float()
            loss = self.criteria(y_hat, label_onehot)
        else:
            loss = self.criteria(y_hat, label)
        self.val_acc(y_hat, label)
        self.val_vra(
            VRA(
                y_hat,
                label,
                L=1 / min(Cifar10DataModule._PREPROCESSING_PARAMS["img_std"]),
                eps=36 / 255,
                last_layer_type=self.last_layer_type,
            )
        )  # L is 1 / max std of imagenet
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
        """if batch_idx == 0:
            dict_dump_all = {'img': img, 'label': label, 'y_hat': y_hat}
            prev_layer = 'img'
            for name in self.model._modules.keys():
                out = self.model._modules[name](dict_dump_all[prev_layer])
                dict_dump_all[name] = out
                prev_layer = name
            torch.save(dict_dump_all, "val_dump.pth")"""

        return loss

    def optimizer_step(
        self,
        epoch: int,
        batch_idx: int,
        optimizer,
        optimizer_closure,
    ) -> None:

        optimizer.step(closure=optimizer_closure)
        # optimizer.zero_grad()
        self.lr_scheduler.step()

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
        self.lr_scheduler = None  # ScheduleFree
        init_lr = 1e-5
        optimizer = Adam(self.parameters(), lr=init_lr, weight_decay=0)

        self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.max_epochs
            * 196,  # bi GPU 98,  # ,  # number of steps for one full cosine cycle
            # 196 batches or 196/world_size
            eta_min=init_lr / 1000.0,  # min learning rate at the end of schedule
        )
        """
        optimizer = schedulefree.AdamWScheduleFree(
            self.parameters(), lr=1e-4, weight_decay=0
        )
        optimizer.train()
        self.hparams["lr"] = optimizer.param_groups[0]["lr"]
        """

        return optimizer


def train(use_wandb, save_model, conf):

    # evaluate_saved_model(conf)
    classification_module = ClassificationLightningModule(
        num_classes=10, classif_args=conf
    )
    # summary(classification_module.model.eval(), input_size=(2,3, 32, 32))
    # if use_wandb:
    #
    #
    #     wandb.init(project="lipschitz_cifar10", config=classification_module.model.__dict__)
    data_module = Cifar10DataModule()
    logger = None
    if use_wandb:
        wandb_logger = WandbLogger(project="lipschitz_cifar10", log_model=False)
        wandb_logger.experiment.config.update(conf)
        logger = [wandb_logger]
    # checkpoint_callback = pl_callbacks.ModelCheckpoint(
    #     monitor="loss",
    #     mode="min",
    #     save_top_k=1,
    #     save_last=True,
    #     dirpath=f"./checkpoints/{wandb_logger.experiment.dir}",
    # )

    lr_logger = LearningRateMonitor(logging_interval="epoch")
    trainer = Trainer(
        accelerator="gpu",
        devices=1,  # GPUs per node
        num_nodes=1,  # Number of nodes
        strategy="ddp",  # Distributed strategy
        precision=32,  # "bf16-mixed",
        max_epochs=conf["epochs"],
        enable_model_summary=True,
        enable_checkpointing=False,
        logger=logger,
        # logger=False,
        callbacks=[
            lr_logger,
            # pl_callbacks.LearningRateFinder(max_lr=0.05),
            # checkpoint_callback,
        ],
    )
    summary(classification_module.model.eval(), input_size=(1, 3, 32, 32))

    trainer.fit(classification_module, data_module)
    # save the model
    classification_module.model.eval()
    if save_model:
        torch.save(classification_module.model.state_dict(), conf["conf_name"] + ".pth")

    evaluate_saved_model(conf, classification_module)
    evaluate_saved_model(conf)

    if use_wandb:
        wandb.finish()


def evaluate_saved_model(conf, model=None):

    if model is None:
        classification_module = ClassificationLightningModule(
            num_classes=10, classif_args=conf
        )
        weights = torch.load(conf["conf_name"] + ".pth")
        classification_module.model.load_state_dict(
            torch.load(conf["conf_name"] + ".pth")
        )
    else:
        classification_module = model
    classification_module.model.to("cuda")
    classification_module.model.eval()
    data_module = Cifar10DataModule()
    val_datataloader = data_module.val_dataloader()
    val_acc = torchmetrics.Accuracy(task="multiclass", num_classes=10).to("cuda")
    val_vra = torchmetrics.MeanMetric().to("cuda")

    with torch.no_grad():
        for batch_idx, batch in enumerate(val_datataloader):
            # batch = batch
            classification_module.model.eval()
            img, label = batch
            img = img.to("cuda")
            label = label.to("cuda")
            y_hat = classification_module.model(img)

            """if batch_idx == 0:
                ref_dict_dump_all = torch.load("val_dump.pth")
                dict_dump_all = {'img': img, 'label': label, 'y_hat': y_hat}
                print("img diff norm", torch.norm(img - ref_dict_dump_all['img']))
                print("y_hat diff norm", torch.norm(y_hat - ref_dict_dump_all['y_hat']))
                prev_layer = 'img'
                for name in classification_module.model._modules.keys():
                    out = classification_module.model._modules[name](dict_dump_all[prev_layer])
                    dict_dump_all[name] = out
                    prev_layer = name
                    print("layer ",name," diff norm", torch.norm(out - ref_dict_dump_all[name]))"""

            val_acc(y_hat, label)
            val_vra(
                VRA(
                    y_hat,
                    label,
                    L=1 / min(Cifar10DataModule._PREPROCESSING_PARAMS["img_std"]),
                    eps=36 / 255,
                    last_layer_type=classification_module.last_layer_type,
                )
            )  # L is 1                                  / max std of imagenet

    print("Validation Accuracy:", val_acc.compute().item())
    print("Validation VRA:", val_vra.compute().item())


if __name__ == "__main__":
    confs = get_confs()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--setting",
        type=str,
        default="stagedcnn_lipbn_robust",
        help=f"The setting to use for training. Can be {confs.keys()}.",
    )
    args = parser.parse_args()
    conf = confs[args.setting]
    use_wandb = True
    save_model = True
    if wandb is None:
        use_wandb = False
    print("setting", args.setting)
    print("conf", conf)
    train(use_wandb, save_model, conf)
