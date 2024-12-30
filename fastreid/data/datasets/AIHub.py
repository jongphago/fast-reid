import re
import glob
import warnings
import os.path as osp


from fastreid.data.datasets import DATASET_REGISTRY
from fastreid.data.datasets.bases import ImageDataset


@DATASET_REGISTRY.register()
class AIHub(ImageDataset):
    _junk_pids = []
    dataset_dir = ""
    dataset_url = ""
    dataset_name = "aihub"

    def __init__(self, root="datasets", **kwargs):
        self.root = root
        self.dataset_dir = osp.join(self.root, self.dataset_dir)

        # allow alternative directory structure
        self.data_dir = self.dataset_dir
        data_dir = osp.join(self.data_dir, "AIHub")
        if osp.isdir(data_dir):
            self.data_dir = data_dir
        else:
            warnings.warn(
                "The current data structure is deprecated. Please "
                'put data folders such as "bounding_box_train" under '
                '"AIHub".'
            )

        self.train_dir = osp.join(self.data_dir, "bounding_box_train")
        self.query_dir = osp.join(self.data_dir, "query")
        self.gallery_dir = osp.join(self.data_dir, "bounding_box_test")
        self.extra_gallery_dir = osp.join(self.data_dir, "images")

        required_files = [
            self.data_dir,
            self.train_dir,
            self.query_dir,
            self.gallery_dir,
        ]
        self.check_before_run(required_files)

        train = lambda: self.process_dir(self.train_dir)
        query = lambda: self.process_dir(self.query_dir, is_train=False)
        gallery = lambda: self.process_dir(self.gallery_dir, is_train=False)

        super(AIHub, self).__init__(train, query, gallery, **kwargs)

    def process_dir(self, dir_path, is_train=True):
        img_paths = glob.glob(osp.join(dir_path, "*.jpg"))
        pattern = re.compile(r"([-\d]+)_c(\d)")

        data = []
        for img_path in img_paths:
            pid, camid = map(int, pattern.search(img_path).groups())
            if pid == -1:
                continue  # junk images are just ignored
            assert 0 <= pid <= 289  # pid == 0 means background
            assert 1 <= camid <= 16
            camid -= 1  # index starts from 0
            if is_train:
                pid = self.dataset_name + "_" + str(pid)
                camid = self.dataset_name + "_" + str(camid)
            data.append((img_path, pid, camid))

        return data
