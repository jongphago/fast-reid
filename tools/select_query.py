import os
import shutil
import random
from tqdm import tqdm
from pathlib import Path
from collections import defaultdict


def create_query_data(test_file_list, query_dir):
    """
    Query 데이터 생성 함수.

    Args:
        test_file_list (dict[Path]): 원본 Gallery 데이터 경로.
        query_dir (str): Query 데이터를 저장할 경로.
    """
    if not os.path.exists(query_dir):
        os.makedirs(query_dir)

    # Gallery 디렉토리에서 ID별로 이미지를 그룹화
    id_to_images = defaultdict(list)

    for file_path in test_file_list:
        pid = int(file_path.stem[:4])
        id_to_images[pid].append(file_path)

    # 각 ID에서 Query 이미지를 선택하여 Query 디렉토리로 이동
    for pid, images in tqdm(id_to_images.items()):
        if len(images) < 3:
            num_query_per_id = 1
        elif len(images) < 10:
            num_query_per_id = 3
        elif len(images) < 100:
            num_query_per_id = 5
        else:  # 이미지 수가 많은 경우
            num_query_per_id = 10
        query_images = random.sample(images, num_query_per_id)
        for img in query_images:
            dst_path = query_dir / f"{img.stem}.jpg"
            shutil.copy(img, dst_path)
    print(f"Query 데이터가 {query_dir}에 생성되었습니다.")


# 예제 실행
data_root = Path("/home/jongphago/Share/projects/fast-reid/datasets/AIHub")
test_dir = data_root / "bounding_box_test"
query_dir = data_root / "_query"
test_file_list = sorted(list(test_dir.glob("*.jpg")))
test_pid_list = {int(file_path.stem[:4]) for file_path in test_file_list}
create_query_data(test_file_list, query_dir)
