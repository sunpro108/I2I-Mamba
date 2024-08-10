from pathlib import Path
from PIL import Image
import torch.utils.data as data
from torchvision import transforms
import torchvision.transforms.functional as tf

from data.base_dataset import BaseDataset, get_transform



class IHarmony4Dataset(BaseDataset):
    """A template dataset class for you to implement custom datasets."""

    def initialize(self, opt):
        """Initialize this dataset class.
        Parameters:
        A few things can be done here.
        - save the options (have been done in BaseDataset)
        - get image paths and meta information of the dataset.
        - define the image transformation.
        """
        # save the option and dataset root
        # super().__init__()
        self.dataset_dir = Path(opt.dataroot)  
        self.image_paths, self.mask_paths, self.gt_paths = [], [], []
        self.is_train = opt.is_for_train # todo
        image_size = opt.loadSize
        self.train_file = None
        self._load_images_paths()
        self.transform_rgb = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5,0.5])
        ])
        self.transform_mask = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor() 
        ]) 

    def _load_images_paths(
        self,
    ):
        file_name = "IHD_train.txt" if self.is_train else "IHD_test.txt"
        self.trainfile = str(self.dataset_dir / file_name)
        with open(self.trainfile, "r") as f:
            for line in f.readlines():
                line = line.rstrip()
                # line = line.replace("jpg", "png")
                name_parts = line.split("_")
                mask_path = line.replace("composite_images", "masks")
                mask_path = mask_path.replace(("_" + name_parts[-1]), ".png")
                gt_path = line.replace("composite_images", "real_images")
                gt_path = gt_path.replace(
                    "_" + name_parts[-2] + "_" + name_parts[-1], ".jpg"
                )
                self.image_paths.append(str(self.dataset_dir / line))
                self.mask_paths.append(str(self.dataset_dir / mask_path))
                self.gt_paths.append(str(self.dataset_dir / gt_path))

    def __getitem__(self, index):
        comp = Image.open(self.image_paths[index]).convert("RGB")
        real = Image.open(self.gt_paths[index]).convert("RGB")
        mask = Image.open(self.mask_paths[index]).convert("1")

        comp = self.transform_rgb(comp)
        real = self.transform_rgb(real)
        mask = self.transform_mask(mask)
        comp = self._compose(comp, mask, real)
        return {
            "comp": comp,
            "mask": mask,
            "real": real,
            "img_path": self._simplify_filepath(self.image_paths[index]),
        }

    def __len__(self):
        """Return the total number of images."""
        return len(self.image_paths)

    def _compose(self, fore, mask, back):
        return fore * mask + back * (1 - mask)

    def _simplify_filepath(self, img_path):
        _names = img_path.split('/')
        dataset_name = _names[-3]
        file_name = _names[-1].split('.')[0]
        return f"{dataset_name}/{file_name}"

    def name(self):
        return "iHarmony4"