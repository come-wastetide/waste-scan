import scissors2
from scissors2 import MagicScissors

data = MagicScissors(
    dataset_size=400,
    min_objects_per_image=4,
    max_objects_per_image=21,
    min_size_variance=0.25,
    max_size_variance=0.4,
    annotate_occlusion=0,
    working_dir="./",
    upload_to_roboflow=False,
    roboflow_api_key="",
    roboflow_workspace="",
    roboflow_project="",
)

data.download_objects_of_interest_from_roboflow(
    url="https://universe.roboflow.com/wastetide/objects-of-interest-public/2"
)

data.download_backgrounds_from_roboflow(
    url="https://app.roboflow.com/wastetide/backgrounds-scswq/1"
)


data.generate_dataset()

