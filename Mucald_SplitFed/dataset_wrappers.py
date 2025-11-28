# dataset_wrappers.py
import os
from torch.utils.data import Dataset

class WithFilenames(Dataset):
    def __init__(self, base_ds, img_dir=None):
        self.base = base_ds
        self.img_dir = img_dir

        # Try to discover how the base dataset stores image paths
        # Adjust these heuristics to your dataset classes if needed:
        if hasattr(base_ds, "img_paths"):
            self._paths = list(base_ds.img_paths)
        elif hasattr(base_ds, "images"):
            self._paths = list(base_ds.images)
        elif hasattr(base_ds, "samples"):
            # e.g. torchvision ImageFolder-like
            # samples is usually [(path, class), ...]
            self._paths = [p if isinstance(p, str) else p[0] for p in base_ds.samples]
        elif hasattr(base_ds, "files"):
            self._paths = list(base_ds.files)
        else:
            # Fallback: we can only synthesize names from indices
            self._paths = [str(i) for i in range(len(base_ds))]

    def __len__(self):
        return len(self.base)

    def _basename(self, idx):
        p = self._paths[idx] if idx < len(self._paths) else str(idx)
        b = os.path.basename(str(p)).strip().lower()
        return b

    def __getitem__(self, idx):
        data = self.base[idx]
        # Expect (img, mask), optionally more fields:
        if isinstance(data, (list, tuple)) and len(data) >= 2:
            img, mask = data[0], data[1]
        else:
            raise RuntimeError("Base dataset must return (image, mask) or similar tuple")

        name = self._basename(idx)
        return img, mask, name
