import os
import json
import logging
from tqdm import tqdm
from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scenario_index_dict import train_scenario_index_dict, test_scenario_index_dict

format = "%(asctime)s [%(process)d|%(thread)d](%(funcName)s:%(lineno)d): %(message)s"
logging.basicConfig(format=format, level=logging.INFO, datefmt="%H:%M:%S")


# Utils
def cvt_color(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def read_img(img_dir: str) -> np.ndarray:
    img_dir = img_dir.as_posix() if isinstance(img_dir, Path) else img_dir
    return cv2.imread(img_dir)


def draw(image, is_convert=True):
    if not isinstance(image, np.ndarray):
        image = read_img(image)
    if is_convert:
        image = cvt_color(image)
    plt.imshow(image)
    plt.axis("off")
    plt.show()


class LabelDict:
    def __init__(self, label_dict: dict):
        self.label_dict = label_dict

    def get_tlbr(self) -> list:
        raise NotImplementedError


class AIHubLabelDict(LabelDict):
    def __init__(self, label_dict: dict):
        super(AIHubLabelDict, self).__init__(label_dict)

        self.track_id = int(self.label_dict["track_id"])
        self.person_id = int(self.label_dict["attributes"][0]["pid"])

    def get_tlbr(self) -> list:
        position_dict = self.label_dict["position"][0]
        x0, y0 = position_dict["x"], position_dict["y"]
        x1, y1 = x0 + position_dict["width"], y0 + position_dict["height"]
        tlbr = list(map(int, [x0, y0, x1, y1]))
        return tlbr

    def get_target_file_name(self, camera_index, frame_id, bbox_id):
        return f"{self.person_id:04d}_c{camera_index:d}s{self.track_id}_{frame_id:06d}_{bbox_id:02d}.jpg"


class Generator:
    def __init__(self, dataset_path: str, scenario_index: int, camera_index: int):
        self.dataset_path = (
            dataset_path if isinstance(dataset_path, Path) else Path(dataset_path)
        )
        self.scenario_index = scenario_index
        self.camera_index = camera_index
        self.initialize_label_dict()
        self.initialize_image_dict()
        assert len(self.image_path_dict) == 321
        assert all(
            key in self.label_path_dict.keys() for key in self.image_path_dict.keys()
        )

    def initialize_label_dict(self):
        raise NotImplementedError

    def initialize_image_dict(self):
        raise NotImplementedError

    def get_label_path(self, frame_id: int, as_posix=False) -> Path:
        raise NotImplementedError

    def get_image_path(self, frame_id: int, as_posix=False) -> Path:
        raise NotImplementedError

    def get_path(self, frame_id, as_posix=False):
        return self.get_label_path(frame_id, as_posix), self.get_image_path(
            frame_id, as_posix
        )

    def get_label(self, frame_id):
        label_path = self.get_label_path(frame_id)
        with open(label_path, "r") as f:
            label = json.load(f)
        return label

    def get_image(self, frame_id):
        image_path = self.get_image_path(frame_id)
        return read_img(image_path)

    def get_image_label_pair(self, frame_id, return_objects=True):
        if return_objects:
            return self.get_image(frame_id), self.get_label(frame_id)["objects"]
        return self.get_image(frame_id), self.get_label(frame_id)


class AIHubTrainDatasetGenerator(Generator):
    def __init__(self, dataset_path: str, scenario_index: int, camera_index: int):
        super(AIHubTrainDatasetGenerator, self).__init__(
            dataset_path, scenario_index, camera_index
        )

    def initialize_label_dict(self):
        t_index = 1 if self.scenario_index < 20 else 2
        label_sub_path = f"라벨링데이터/TL{t_index}/시나리오{self.scenario_index:02d}/카메라{self.camera_index:02d}"
        label_path_list = sorted(
            list((self.dataset_path / label_sub_path).glob("*.json"))
        )
        self.label_path_dict = {
            int(filename.stem[-4:]): filename for filename in label_path_list
        }

    def initialize_image_dict(self):
        image_sub_path = (
            f"frames/시나리오{self.scenario_index:02d}/카메라{self.camera_index:02d}"
        )
        image_path_list = sorted(
            list((self.dataset_path / image_sub_path).glob("*.jpg"))
        )
        self.image_path_dict = {
            int(filename.stem[-4:]): filename for filename in image_path_list
        }

    def get_label_path(self, frame_id: int, as_posix=False) -> Path:
        label_path = self.label_path_dict[frame_id]
        if not label_path.exists():
            raise FileNotFoundError(f"{label_path} does not exist.")
        return label_path

    def get_image_path(self, frame_id: int, as_posix=False) -> Path:
        image_path = self.image_path_dict[frame_id]
        if not image_path.exists():
            raise FileNotFoundError(f"{image_path} does not exist.")
        return image_path if not as_posix else image_path.as_posix()


class AIHubTestDatasetGenerator(AIHubTrainDatasetGenerator):
    def __init__(self, dataset_path: str, scenario_index: int, camera_index: int):
        super(AIHubTestDatasetGenerator, self).__init__(
            dataset_path, scenario_index, camera_index
        )

    def initialize_label_dict(self):
        label_sub_path = f"라벨링데이터/VL1/시나리오{self.scenario_index:02d}/카메라{self.camera_index:02d}"
        label_path_list = sorted(
            list((self.dataset_path / label_sub_path).glob("*.json"))
        )
        self.label_path_dict = {
            int(filename.stem[-4:]): filename for filename in label_path_list
        }


def generate(
    Generator,
    dataset_path,
    train_scenario_index_dict,
    target_sub_dir,
):
    total_iterations = sum(
        len(camera_indices) * 321
        for camera_indices in train_scenario_index_dict.values()
    )
    with tqdm(total=total_iterations) as pbar:
        for scenario_index, camera_indices in train_scenario_index_dict.items():
            for camera_index in camera_indices:
                # logging.info(f"Scenario: {scenario_index}, Camera: {camera_index}")
                dataset_generator = Generator(
                    dataset_path, scenario_index, camera_index
                )
                num_image_path = len(dataset_generator.image_path_dict)
                # logging.info(f"Num Images: {num_image_path}")
                for frame_id in dataset_generator.image_path_dict.keys():
                    pbar.update(1)
                    image, labels = dataset_generator.get_image_label_pair(frame_id)
                    height, width, _ = image.shape
                    if labels[0]["label"] == "void":
                        continue
                    for bbox_id, label in enumerate(labels):
                        if label["label"] == "blackout":
                            continue
                        label_dict = AIHubLabelDict(label)
                        tlbr = label_dict.get_tlbr()
                        target_file_name = label_dict.get_target_file_name(
                            camera_index, frame_id, bbox_id
                        )
                        x0, y0, x1, y1 = tlbr
                        x0, y0, x1, y1 = (
                            max(0, x0),
                            max(0, y0),
                            min(width, x1),
                            min(height, y1),
                        )

                        crop_image = image[y0:y1, x0:x1]
                        target_file_name = (
                            dataset_generator.dataset_path
                            / target_sub_dir
                            / target_file_name
                        )
                        if not crop_image.size:
                            continue
                        cv2.imwrite(target_file_name, crop_image)


if __name__ == "__main__":
    data_root = "/home/jongphago/Share/datasets/148.멀티센서 동선 추적 데이터/01.데이터"
    data_root = Path(data_root)
    generate(
        AIHubTestDatasetGenerator,
        data_root / "2.Validation",
        test_scenario_index_dict,
        "bounding_box_test",
    )
    generate(
        AIHubTrainDatasetGenerator,
        data_root / "1.Training",
        train_scenario_index_dict,
        "bounding_box_train",
    )
