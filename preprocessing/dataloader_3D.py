import tensorflow as tf
from pathlib import Path
import numpy as np
from PIL import Image
import re


def parse_img(
    img_dir: Path,
    num_zslices: int = 17,
    num_channels: int = 6,  # 5 or 6 for images, for masks set to 14
    res: int = 512,
    start_ch=0,  # 0 for imgs, for masks set to 5
):
    """ Parses a volumetric image or mask from a given file path and returns an numpy array
            Args:
                img_dir: Pathlib path pointing to folder containing all image slices to load in
                num_zslices: number of image slices the dataloader should expect
                num_channels: number of channels for the image to load
                res: resolution(xy) to downscale volumetric image to
                start_ch: start channel index
        """
    if not img_dir.exists() or not img_dir.is_dir():
        raise ValueError("img_dir didn't exist or wasn't a directory")
    num_files = len([file for file in img_dir.rglob("*.tif")])
    # print(f" {num_files} files, {num_zslices} slices, {num_channels} channels")
    # folder needs to contain 17 z-slices @ 6 channels each
    if num_files != num_zslices * num_channels:
        raise ValueError(
            f"Found {num_files} files in img_dir, but expected {num_zslices*num_channels}"
        )
    # extract slices for each channel
    tensor_slices = []
    for i in range(start_ch, start_ch + num_channels):
        tmp = {
            str(re.split(r"[_.\s]\s*", img_path.parts[-1])[-2]): img_path
            for img_path in img_dir.rglob(f"*_C*{i}_*.tif")
        }
        if len(tmp.keys()) != num_zslices:
            raise ValueError(
                f"expected {num_zslices} z-slices for channel {i}, but found {len(tmp.keys())}"
            )
        sorted_tmp = {k: v for k, v in sorted(tmp.items())}
        tmp_imgs = {
            z_key: np.array(Image.open(img_path).resize((res, res)))
            for z_key, img_path in sorted_tmp.items()
        }
        tensor_slices.append(list(tmp_imgs.values()))
    out = tf.transpose(tf.stack(tensor_slices), [2, 3, 1, 0])
    return np.array(out)


class ImmunoDataloader3D:
    """Takes a path string and returns ready-to-use np-arrays for training
        Expects: - One path to root folder containing two subfolders: 'Control' and 'COXKO'
                 - a separate .png file for each color channel (6 channels in total per image)
    """

    def __init__(
        self,
        path: str,
        num_channels: int = 6,
        res: int = 512,
        start_ch: int = 0,
        verbose: bool = True,
    ):
        self.images, self.labels = [], []
        self.load_data(
            path_str=path,
            res=res,
            num_channels=num_channels,
            start_ch=start_ch,
            verbose=verbose,
        )

    def load_data(
        self, path_str: str, res: int, num_channels: int, start_ch: int, verbose: bool
    ):
        """ Loads dataset from given path:
            Args:
                path_str: path (in string format) that points to dataset location
                res: resolution(xy) to downscale images to
                num_channels: number of channels for each volumetric image
                start_ch: starting channel index e.g 1
                verbose: whether to print progress
        """
        path = Path(path_str)
        print(str(path))
        if path.exists():
            control_path = path / "Control"
            positive_path = path / "COXKO"
            if control_path.exists() and positive_path.exists():
                # control images
                control_images = []
                print(
                    f"found {len(list(control_path.glob('*')))} control images"
                ) if verbose else 0
                for i, p in enumerate(control_path.iterdir()):
                    print(f"loading image {i+1} from control: {p}") if verbose else 0
                    try:
                        img = parse_img(
                            p, res=res, num_channels=num_channels, start_ch=start_ch
                        )
                    except ValueError as e:
                        print(f"something went wrong loading in {p}: {e}")
                    control_images.append(img)
                # positive images
                positive_images = []
                print(
                    f"found {len(list(positive_path.glob('*')))} control images"
                ) if verbose else 0
                for i, p in enumerate(positive_path.iterdir()):
                    print(f"loading image {i+1} from positive: {p}")
                    try:
                        img = parse_img(
                            p, res=res, num_channels=num_channels, start_ch=start_ch
                        )
                    except ValueError as e:
                        print(f"something went wrong loading in {p}: {e}")
                    positive_images.append(img)

                self.images = np.array(control_images + positive_images)
                self.path_names = [i for i in positive_path.iterdir()] + [
                    i for i in control_path.iterdir()
                ]
                self.labels = np.concatenate(
                    (np.zeros(len(control_images)), np.ones(len(positive_images)))
                )

        else:
            raise FileNotFoundError(f"{path} not found")
