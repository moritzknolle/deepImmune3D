import pathlib
from pathlib import Path
import numpy as np
from PIL import Image


class ImmunoDataloader:
    """Takes a path string and returns ready-to-use np-arrays for training
        Expects: - One path to root folder containing two subfolders: 'Control' and 'Positive'
                 - a separate .png file for each color channel (6 channels in total per image)
    """

    def __init__(self, path: str, res: int, verbose: bool):
        self.load_data(path, res, verbose)

    def load_data(self, path_str: str, res: int, verbose):
        """ Loads dataset from given path:
            Args:
                path_str: path (in string format) that points to dataset location
                res: resolution(xy) to downscale images to
                verbose: whether to print progress
        """
        path = Path(path_str)
        if path.exists():
            control_path = path / "Control"
            positive_path = path / "Positive"
            if control_path.exists() and positive_path.exists():
                # control images
                control_images = []
                for i, p in enumerate(control_path.iterdir()):
                    inner = []
                    for img in sorted(p.rglob("C*")):
                        print("processing", img) if verbose else 0
                        inner.append(
                            np.array((Image.open(str(img)).resize((res, res))))[..., 0]
                        )
                    print(f"loaded image {i+1} from control")
                    control_images.append(np.stack(inner))
                # positive images
                positive_images = []
                for i, p in enumerate(positive_path.iterdir()):
                    inner = []
                    for img in sorted(p.rglob("C*")):
                        print("processing:", img) if verbose else 0
                        inner.append(
                            np.array((Image.open(str(img)).resize((res, res))))[..., 0]
                        )
                    print(f"loaded image {i+1} from positive")
                    positive_images.append(np.stack(inner))

                self.images = np.moveaxis(
                    np.array(control_images + positive_images), 1, 3
                )
                self.labels = np.concatenate(
                    (np.zeros(len(control_images)), np.ones(len(positive_images)))
                )

        else:
            raise FileNotFoundError()
