import torch
import albumentations
from albumentations.pytorch import ToTensor


def TransformerSeg(image=None, mask=None):
    img_trans = albumentations.Compose([
        albumentations.Normalize(mean=[0.798, 0.621, 0.841], std=[0.125, 0.228, 0.089]),
        ToTensor(),
    ])
    result = img_trans(image=image)
    if mask is not None:
        result['mask'] = torch.tensor(mask, dtype=torch.long)

    return result

