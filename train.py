import argparse
from pathlib import Path
import os
import wandb

import numpy as np
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from torchinfo import summary

from multimodal.multimodal_data_module import MultiModalDataModule
from multimodal.multimodal_saycam_data_module import MultiModalSAYCamDataModule

from multimodal.data_modules import MultiModalSAYCamDataModuleBabyFM, MultiModalTripletDataModule

from multimodal.multimodal import VisionEncoder, TextEncoder, MultiModalModel, LanguageModel
from multimodal.multimodal_lit import MultiModalLitModel, TripletLitModel, TouchClassifierLitModel


def _setup_parser():
    """Set up Python's ArgumentParser with data, model, trainer, and other arguments."""
    parser = argparse.ArgumentParser()

    # add trainer specific arguments
    trainer_parser = pl.Trainer.add_argparse_args(parser)
    trainer_parser._action_groups[1].title = "Trainer Args"  # pylint: disable=protected-access
    parser = argparse.ArgumentParser(add_help=False, parents=[trainer_parser])

    # get data, model and litmodel specific arguments
    data_group = parser.add_argument_group("Data Args")
    MultiModalDataModule.add_to_argparse(data_group)
    MultiModalSAYCamDataModule.add_additional_to_argparse(data_group)
    
    MultiModalSAYCamDataModuleBabyFM.add_additional_to_argparse(data_group)

    model_group = parser.add_argument_group("Model Args")
    VisionEncoder.add_to_argparse(model_group)
    TextEncoder.add_to_argparse(model_group)
    MultiModalModel.add_to_argparse(model_group)
    LanguageModel.add_to_argparse(model_group)

    lit_model_group = parser.add_argument_group("LitModel Args")
    MultiModalLitModel.add_to_argparse(lit_model_group)
    TripletLitModel.add_to_argparse(lit_model_group)
    TouchClassifierLitModel.add_to_argparse(lit_model_group)

    parser.add_argument("--exp_name", type=str, default="multimodal_test",
                        help="experiment name for logging")
    parser.add_argument("--triplet", action="store_true",
                        help="use triplet mode with audio and touch modalities instead of regular mode")
    parser.add_argument("--seed", type=int, default=0,
                        help="random seed for everything")
    parser.add_argument("--resume_ckpt", type=Path, default=None,
                        help="path to the checkpoint to resume from; if it's "
                             "\"last\", resume from the last checkpoint.")
    parser.add_argument("--pretrained_ckpt", type=Path, default=None,
                        help="path to pretrained checkpoint to load vision and audio encoders from (touch encoder stays random)")
    parser.add_argument("--train_data_path", type=str, default="/projectnb/ivc-ml/maxwh/code/labeling_effort/Dataset_v2/training_data_with_images.json",
                        help="Path to training data JSON file")
    parser.add_argument("--dino_ckpt", type=str, default=None,
                        help="Path to DINO checkpoint for vision encoder initialization")
    parser.add_argument("--checkpoint_dir", type=str, default=None,
                        help="Directory to save checkpoints")

    return parser


def _is_raw_vit_state_dict(sd):
    """Heuristic: a bare DINO/ViT backbone state_dict has top-level keys like
    cls_token / pos_embed / patch_embed.* / blocks.* (no lit-model prefix)."""
    markers = {"cls_token", "pos_embed"}
    return any(k in markers or k.startswith("patch_embed.") or k.startswith("blocks.")
               for k in sd.keys())


def _extract_vit_backbone(ckpt):
    """Return a bare ViT-backbone state_dict from a checkpoint of one of:
      - DINO training checkpoint with 'teacher'/'student' top-level keys
      - bare ViT state_dict (returned as-is)
      - None if it doesn't look like a ViT-only checkpoint
    """
    if isinstance(ckpt, dict) and ("teacher" in ckpt or "student" in ckpt):
        sd = ckpt.get("teacher", ckpt.get("student"))
        # strip the DataParallel `module.` wrapper, then the multicrop `backbone.` wrapper
        sd = {k[len("module."):] if k.startswith("module.") else k: v for k, v in sd.items()}
        # keep only backbone params; drop the DINO projection head used during SSL pre-training
        sd = {k[len("backbone."):]: v for k, v in sd.items() if k.startswith("backbone.")}
        return sd
    if isinstance(ckpt, dict) and _is_raw_vit_state_dict(ckpt):
        return ckpt
    return None


def _load_pretrained_checkpoint(lit_model, ckpt_path):
    """Load a pretrained checkpoint into lit_model. Supports:
      - Lightning checkpoints (lit-model-prefixed state_dict)
      - bare ViT state_dicts (e.g. models/dino_teacher_pretrained.pth)
      - DINO training checkpoints (e.g. models/dino_pretrained.pth)
    For the ViT-only formats, weights are loaded directly into
    lit_model.vision_encoder.model — the newly-added head linear layer
    has no counterpart in the checkpoint and stays random-initialized.
    """
    print(f"Loading full model from: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu")

    vit_sd = _extract_vit_backbone(ckpt)
    if vit_sd is not None:
        print(f"Detected ViT-only checkpoint ({len(vit_sd)} keys); "
              f"loading into vision_encoder.model")
        missing, unexpected = lit_model.vision_encoder.model.load_state_dict(
            vit_sd, strict=False)
        # The head linear projection is added after the DINO backbone load and is
        # random-initialized by design — exclude it from the "missing" report.
        missing = [k for k in missing if not k.startswith("head.")]
    else:
        state_dict = ckpt.get("state_dict", ckpt)
        missing, unexpected = lit_model.load_state_dict(state_dict, strict=False)

    print(f"Loaded checkpoint.")
    print(f"Missing keys: {len(missing)}")
    for k in missing:
        print(f"  - {k}")
    print(f"Unexpected keys: {len(unexpected)}")
    for k in unexpected:
        print(f"  - {k}")


def main():
    # parse args
    parser = _setup_parser()
    # parser.print_help()
    args = parser.parse_args()

    # checkpoint paths
    if args.checkpoint_dir is not None:
        ckpt_dir = Path(args.checkpoint_dir)
    else:
        ckpt_dir = Path(args.exp_name)
    
    # set random seed
    pl.seed_everything(args.seed)

    if args.triplet:
        print("Using Triplet data module with audio and touch modalities")
        DataModuleClass = MultiModalTripletDataModule
    else:
        print("Using regular mode with audio modality only")
        DataModuleClass = MultiModalSAYCamDataModuleBabyFM

  
    data = DataModuleClass(args)
    vocab = data.read_vocab()

    vision_encoder = VisionEncoder(args=args)
  
    # For triplet mode, create separate audio and touch encoders with their own parameters
    if args.triplet:
        audio_encoder = TextEncoder(
            vocab, image_feature_map_dim=vision_encoder.last_cnn_out_dim, args=args)
        touchvocab = {}
        for i in range(args.num_touch_classes):
            touchvocab[i] = i
        touch_encoder = TextEncoder(
            touchvocab, image_feature_map_dim=vision_encoder.last_cnn_out_dim, nhot=True, args=args)
    
        print("Touch encoder initialized randomly (not loaded from checkpoint)")
        lit_model = TripletLitModel(vision_encoder, audio_encoder, touch_encoder, args)
        if args.pretrained_ckpt is not None:
            _load_pretrained_checkpoint(lit_model, args.pretrained_ckpt)

    else:
        text_encoder = TextEncoder(
            vocab, image_feature_map_dim=vision_encoder.last_cnn_out_dim, args=args)
        lit_model = MultiModalLitModel(vision_encoder, text_encoder, args)
        if args.pretrained_ckpt is not None:
            _load_pretrained_checkpoint(lit_model, args.pretrained_ckpt)
    

    checkpoint_callback = ModelCheckpoint(
        dirpath=ckpt_dir,
        filename="{epoch}",
        every_n_epochs=1,
        save_top_k=-1,  #saves all checkpoints 
        save_on_train_epoch_end=True    #no validation loop needed
    )




    # create trainer (with checkpoint and logger if specified)
    if args.logger:

        wandb_logger = WandbLogger(project='BabyFM-CLIP', name=args.exp_name,
                                   log_model=False)
        trainer = pl.Trainer.from_argparse_args(args,
                                                enable_checkpointing=True,
                                                callbacks=[
                                                    checkpoint_callback],
                                                logger=wandb_logger)
    else:
        trainer = pl.Trainer.from_argparse_args(args,
                                                enable_checkpointing=True,
                                                callbacks=[
                                                    checkpoint_callback])

    import multimodal.data_modules as data_modules
    data_modules.TRAIN_DATA_DIR = Path(args.train_data_path)
    
    if args.num_touch_classes is not None:
        data_modules.TOUCH_CLUSTER_KEY = f"touch_cluster_{args.num_touch_classes}"
        print(f"Using touch cluster key: {data_modules.TOUCH_CLUSTER_KEY}")

    trainer.fit(lit_model, data)
  

if __name__ == "__main__":
    main()
