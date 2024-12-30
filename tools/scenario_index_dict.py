import logging
from pathlib import Path

format = "%(asctime)s [%(process)d|%(thread)d](%(funcName)s:%(lineno)d): %(message)s"
logging.basicConfig(format=format, level=logging.INFO, datefmt="%H:%M:%S")


def scenario_index_dict_generator(data_root, sub_dir):
    data_root = data_root / sub_dir
    scenario_index_dict = {}
    for p in sorted(list((data_root / "frames").glob("*"))):
        if not p.is_dir():
            continue
        camera_index_list = []
        
        for sub_p in list(p.glob("*")):
            if not sub_p.is_dir():
                continue
            if len(list(sub_p.glob("*.jpg"))) != 321:
                continue
            camera_index_list.append(int(sub_p.stem[-2:]))
        camera_index_list.sort()
        scenario_index_dict[int(p.stem[-2:])] = camera_index_list
    return scenario_index_dict


train_scenario_index_dict = {
    1: [2, 4, 6],
    10: [2, 4, 6],
    11: [2, 6],
    12: [1, 2, 3, 4, 5, 7, 9, 10, 11, 12, 13, 14, 15, 16],
    13: [2, 4, 6],
    14: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
    15: [1, 2, 4, 5, 6, 7, 9, 10, 11, 12, 15],
    16: [],
    17: [2, 4, 6],
    18: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
    31: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
    32: [1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
    33: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
    34: [1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
    35: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
    36: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
    37: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
    38: [1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 14, 15, 16],
    39: [1, 3, 4, 5, 7, 8, 10, 11, 12, 13, 16],
}

test_scenario_index_dict = {
    19: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
    42: [1, 2, 3, 4, 5, 6, 7, 8, 11, 12, 13, 14, 15],
    43: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
}


if __name__ == "__main__":
    data_root = "/home/jongphago/Share/datasets/148.멀티센서 동선 추적 데이터/01.데이터/"
    data_root = Path(data_root)
    train_scenario_index_dict = scenario_index_dict_generator(data_root, "1.Training")
    test_scenario_index_dict = scenario_index_dict_generator(data_root, "2.Validation")
    logging.info(f"Train Scenario Index Dict: {train_scenario_index_dict}")
    logging.info(f"Test Scenario Index Dict: {test_scenario_index_dict}")
