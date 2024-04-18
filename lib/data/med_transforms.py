from re import L
import numpy as np
from monai import transforms
from torch import scalar_tensor, zero_
from monai.transforms.transform import LazyTransform, MapTransform, RandomizableTransform
import random

class ScaleByMidDim(MapTransform):
    def __init__(self, keys, mode='area', max_size=128,allow_missing_keys=False):
        self.keys = keys
        self.mode = mode  
        self.allow_missing_keys = allow_missing_keys
        self.max_size=max_size

    def __call__(self, data):
        for key in self.keys:
            ct_tensor = data[key]
            # max_dim = max(ct_tensor.shape)
            # scale_factor = self.max_size / max_dim
            sorted_numbers = sorted(ct_tensor.shape[1:])
            scale_factor = self.max_size / sorted_numbers[1]
            new_size = [int(d * scale_factor) for d in ct_tensor.shape[1:]]
            # resizer = transforms.Resized(keys=[key], spatial_size=new_size[1:], 
            resizer = transforms.Resized(keys=[key], spatial_size=new_size,   
                mode=self.mode, allow_missing_keys=self.allow_missing_keys)
            data[key] = resizer(data)[key]
        return data
    
class RandScaleCropdPlusScaleByMidDimSampled(MapTransform):
    def __init__(self, keys, mode='area', max_size=128,allow_missing_keys=False,num_samples=4,max_radio=0.8,min_radio=0.5):
        self.keys = keys
        self.mode = mode  
        self.allow_missing_keys = allow_missing_keys
        self.max_size=max_size
        self.num_samples = num_samples
        self.max_radio=max_radio
        self.min_radio=min_radio
        
    def __call__(self, data):
        outputs = []
        for i in range(self.num_samples):  
            random_number = round(random.uniform(self.min_radio, self.max_radio), 2)
            _data = dict(data)
            for key in self.keys:
                cropper= transforms.RandScaleCropd(keys=[key],roi_scale=random_number) 
                _data[key] = cropper(_data)[key] 
                ct_tensor = _data[key] 
                sorted_numbers = sorted(ct_tensor.shape[1:])
                scale_factor = self.max_size / sorted_numbers[1]
                new_size = [int(d * scale_factor)  
                            for d in ct_tensor.shape[1:]]
                
                resizer = transforms.Resized(keys=[key],  
                                             spatial_size=new_size,
                                             mode=self.mode,  
                                             allow_missing_keys=self.allow_missing_keys)
                _data[key] = resizer(_data)[key]    
                
            outputs.append(_data)
              
        return outputs
    
class ConvertToMultiChannelBasedOnBratsClassesd(transforms.MapTransform):
    """
    Convert labels to multi channels based on brats classes:
    label 1 is the peritumoral edemaf
    label 2 is the GD-enhancing tumor
    label 3 is the necrotic and non-enhancing tumor core
    The possible classes are TC (Tumor core), WT (Whole tumor)
    and ET (Enhancing tumor).

    """

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            result = []
            # merge label 2 and label 3 to construct TC
            result.append(np.logical_or(d[key] == 2, d[key] == 3))
            # merge labels 1, 2 and 3 to construct WT
            result.append(
                np.logical_or(
                    np.logical_or(d[key] == 2, d[key] == 3), d[key] == 1
                )
            )
            # label 2 is ET
            result.append(d[key] == 2)
            d[key] = np.concatenate(result, axis=0).astype(np.float32)
        return d

def get_scratch_train_transforms(args):
    if args.dataset == 'btcv':
        train_transform = transforms.Compose(
            [
                transforms.LoadImaged(keys=["image", "label"]),
                transforms.AddChanneld(keys=["image", "label"]),
                transforms.Orientationd(keys=["image", "label"], axcodes="RAS"),
                transforms.Spacingd(keys=["image", "label"],
                                    pixdim=(args.space_x, args.space_y, args.space_z),
                                    mode=("bilinear", "nearest")),
                transforms.ScaleIntensityRanged(keys=["image"],
                                                a_min=args.a_min,
                                                a_max=args.a_max,
                                                b_min=args.b_min,
                                                b_max=args.b_max,
                                                clip=True),
                transforms.CropForegroundd(keys=["image", "label"], source_key="image"),
                transforms.RandCropByPosNegLabeld(
                    keys=["image", "label"],
                    label_key="label",
                    spatial_size=(args.roi_x, args.roi_y, args.roi_z),
                    pos=1,
                    neg=1,
                    num_samples=args.num_samples,
                    image_key="image",
                    image_threshold=0,
                ),
                transforms.RandFlipd(keys=["image", "label"],
                                    prob=args.RandFlipd_prob,
                                    spatial_axis=0),
                transforms.RandFlipd(keys=["image", "label"],
                                    prob=args.RandFlipd_prob,
                                    spatial_axis=1),
                transforms.RandFlipd(keys=["image", "label"],
                                    prob=args.RandFlipd_prob,
                                    spatial_axis=2),
                transforms.RandRotate90d(
                    keys=["image", "label"],
                    prob=args.RandRotate90d_prob,
                    max_k=3,
                ),
                transforms.RandScaleIntensityd(keys="image",
                                            factors=0.1,
                                            prob=args.RandScaleIntensityd_prob),
                transforms.RandShiftIntensityd(keys="image",
                                            offsets=0.1,
                                            prob=args.RandShiftIntensityd_prob),
                transforms.ToTensord(keys=["image", "label"]),
            ]
        )
    else:
        raise ValueError(f"Only support BTCV transforms for medical images")
    return train_transform

def get_mae_pretrain_transforms(args):
    if args.dataset == 'btcv':
        train_transform = transforms.Compose(
            [

                transforms.LoadImaged(keys=["image"],allow_missing_keys=True),
                transforms.AddChanneld(keys=["image"],allow_missing_keys=True),
                transforms.Orientationd(keys=["image"],
                                        axcodes="RAS",allow_missing_keys=True),
                transforms.Spacingd(keys=["image"],
                                    pixdim=(args.space_x, args.space_y, args.space_z),
                                    mode=("bilinear"),
                                    allow_missing_keys=True),
                transforms.ScaleIntensityRanged(keys=["image"],
                                                a_min=args.a_min,
                                                a_max=args.a_max,
                                                b_min=args.b_min,
                                                b_max=args.b_max,
                                                clip=True),
                transforms.CropForegroundd(keys=["image"], source_key="image",allow_missing_keys=True),
                transforms.SpatialPadd(keys=["image"],spatial_size=(args.roi_x, args.roi_y, args.roi_z), mode='constant'),
                transforms.RandSpatialCropSamplesd(keys=["image"],
                                                    roi_size=(args.roi_x, args.roi_y, args.roi_z), 
                                                    num_samples=args.num_samples),
                transforms.RandFlipd(keys=["image"],
                                    prob=args.RandFlipd_prob,
                                    spatial_axis=0),
                transforms.RandFlipd(keys=["image"],
                                    prob=args.RandFlipd_prob,
                                    spatial_axis=1),
                transforms.RandFlipd(keys=["image"],
                                    prob=args.RandFlipd_prob,
                                    spatial_axis=2),
                transforms.ToTensord(keys=["image"])
            ]
        )
    else:
        raise ValueError(f"Only support BTCV transforms for medical images")
    return train_transform


def get_raw_transforms(args):
    if args.dataset == 'btcv':
        val_transform = transforms.Compose(
            [

                transforms.LoadImaged(keys=["image"]),
                transforms.AddChanneld(keys=["image"]),
                transforms.Orientationd(keys=["image"],
                                        axcodes="RAS"),
                transforms.Spacingd(keys=["image"],
                                    pixdim=(args.space_x, args.space_y, args.space_z),
                                    mode=("bilinear")),
                transforms.ScaleIntensityRanged(keys=["image"],
                                                a_min=args.a_min,
                                                a_max=args.a_max,
                                                b_min=args.b_min,
                                                b_max=args.b_max,
                                                clip=True),
                transforms.CropForegroundd(keys=["image"], source_key="image"),
                transforms.ToTensord(keys=["image"]),
            ]
        )
    else:
        raise ValueError(f"Only support BTCV transforms for medical images")
    return val_transform

class Resize():
    def __init__(self, scale_params):
        self.scale_params = scale_params

    def __call__(self, img):
        scale_params = self.scale_params
        shape = img.shape[1:]
        assert len(scale_params) == len(shape)
        spatial_size = []
        for scale, shape_dim in zip(scale_params, shape):
            spatial_size.append(int(scale * shape_dim))
        transform = transforms.Resize(spatial_size=spatial_size, mode='nearest')
        # import pdb
        # pdb.set_trace()
        return transform(img)

def get_post_transforms(args):
    if args.dataset == 'btcv':
        if args.test:
            post_pred = transforms.Compose([transforms.EnsureType(),
                                            # Resize(scale_params=(args.space_x, args.space_y, args.space_z)),
                                            transforms.AsDiscrete(argmax=True, to_onehot=args.num_classes)])
            post_label = transforms.Compose([transforms.EnsureType(),
                                            # Resize(scale_params=(args.space_x, args.space_y, args.space_z)),
                                            transforms.AsDiscrete(to_onehot=args.num_classes)])
        else:
            post_pred = transforms.Compose([transforms.EnsureType(),
                                            transforms.AsDiscrete(argmax=True, to_onehot=args.num_classes)])
            post_label = transforms.Compose([transforms.EnsureType(),
                                            transforms.AsDiscrete(to_onehot=args.num_classes)])
    return post_pred, post_label