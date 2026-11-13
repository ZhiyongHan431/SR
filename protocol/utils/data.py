import os.path as osp

import PIL.Image as PImage

from dataloader.alignvar_repro import AlignVARPairDataset, LSDIRFFHQDataset


def normalize_01_into_pm1(x):  # normalize x from [0, 1] to [-1, 1] by (x*2) - 1
    return x.add(x).add_(-1)


def build_dataset(
    valid_data_path: str, final_reso: int,
    tokenizer, null_text_ratio, original_image_ratio,
    train_manifest_path: str = 'data/manifests/lsdir_ffhq10k.txt',
    degradation_config: str = 'dataloader/params_realesrgan.yml',
    cfg_mode: str = 'no_cfg',
    synthetic_negative_fraction: float = 0.02,
    synthetic_negative_seed: int = 2025,
):
    """Build only the data adapters; the released VARSR model is unchanged."""
    train_set = LSDIRFFHQDataset(
        manifest_path=train_manifest_path,
        image_size=final_reso,
        degradation_config=degradation_config,
        cfg_mode=cfg_mode,
        synthetic_negative_fraction=synthetic_negative_fraction,
        synthetic_negative_seed=synthetic_negative_seed,
    )
    val_set = AlignVARPairDataset(valid_data_path, image_size=final_reso)
    num_classes = 2
    print(f'[Dataset] {len(train_set)=}, {len(val_set)=}, {num_classes=}')    
    return num_classes, train_set, val_set


def pil_loader(path):
    with open(path, 'rb') as f:
        img: PImage.Image = PImage.open(f).convert('RGB')
    return img


def print_aug(transform, label):
    print(f'Transform {label} = ')
    if hasattr(transform, 'transforms'):
        for t in transform.transforms:
            print(t)
    else:
        print(transform)
    print('---------------------------\n')
