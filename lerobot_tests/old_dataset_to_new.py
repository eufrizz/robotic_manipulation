from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.robot_devices.control_configs import RecordControlConfig
import numpy as np
import argparse
import datasets
from pathlib import Path
from tqdm import tqdm

# cfg = RecordControlConfig(
#     repo_id="eufrizz/lite6_record_scripted_250204",
#     single_task="Pick up the block",
#     root="datasets/lite6_record_scripted_250204",
#     fps=30,
#     video=True,
#     # push_to_hub=False
# )

""" pusht features
From pusht dataset:

{'observation.image': {'dtype': 'video',
  'shape': (96, 96, 3),
  'names': ['height', 'width', 'channel'],
  'video_info': {'video.fps': 10.0,
   'video.codec': 'av1',
   'video.pix_fmt': 'yuv420p',
   'video.is_depth_map': False,
   'has_audio': False}},
 'observation.state': {'dtype': 'float32',
  'shape': (2,),
  'names': {'motors': ['motor_0', 'motor_1']}},
 'action': {'dtype': 'float32',
  'shape': (2,),
  'names': {'motors': ['motor_0', 'motor_1']}},
 'episode_index': {'dtype': 'int64', 'shape': (1,), 'names': None},
 'frame_index': {'dtype': 'int64', 'shape': (1,), 'names': None},
 'timestamp': {'dtype': 'float32', 'shape': (1,), 'names': None},
 'next.reward': {'dtype': 'float32', 'shape': (1,), 'names': None},
 'next.done': {'dtype': 'bool', 'shape': (1,), 'names': None},
 'next.success': {'dtype': 'bool', 'shape': (1,), 'names': None},
 'index': {'dtype': 'int64', 'shape': (1,), 'names': None},
 'task_index': {'dtype': 'int64', 'shape': (1,), 'names': None}}
"""

# features={'action.qpos': {'dtype': 'float32',
#   'shape': (6,)},
#   'action.gripper': {'dtype': 'int8',
#   'shape': (1,)},
#  'observation.state.qpos': {'dtype': 'float32',
#   'shape': (6,)},
#  'observation.state.qvel': {'dtype': 'float32',
#   'shape': (6,)},
#   'observation.state.gripper': {'dtype': 'float32',
#   'shape': (1,)},
#  'observation.ee_pose.pos': {'dtype': 'float32',
#   'shape': (3,)},
# 'observation.ee_pose.quat': {'dtype': 'float32',
#   'shape': (4,)},
# 'observation.ee_pose.vel': {'dtype': 'float32',
#   'shape': (3,)},
# 'observation.ee_pose.ang_vel': {'dtype': 'float32',
#   'shape': (3,)},
#  'observation.images.gripper': {'dtype': 'video',
#   'shape': (240, 320, 3),
#   'names': ['height', 'width', 'channels'],
#   'info': None},
#  'observation.images.side': {'dtype': 'video',
#   'shape': (240, 320, 3),
#   'names': ['height', 'width', 'channels'],
#   'info': None},
#    'episode_index': {'dtype': 'int64', 'shape': (1,), 'names': None},
#  'frame_index': {'dtype': 'int64', 'shape': (1,), 'names': None},
#  'timestamp': {'dtype': 'float32', 'shape': (1,), 'names': None},
#  'reward': {'dtype': 'float32', 'shape': (1,), 'names': None},
#  'index': {'dtype': 'int64', 'shape': (1,), 'names': None},
#  'task_index': {'dtype': 'int64', 'shape': (1,), 'names': None}}



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
                    prog='Lerobot old dataset to new',
                    description='')
    parser.add_argument('input', type=str)
    parser.add_argument('description', type=str)
    parser.add_argument('--out-dir', default='datasets/', type=str)

    args = parser.parse_args()

    old_dataset = datasets.load_from_disk(args.input)
    print(old_dataset)
    dataset_name = Path(args.input).with_suffix('').name
    print(dataset_name)
    # Average fps across all episodes - assumes they are all same start and end time
    fps = len(old_dataset)/(old_dataset[-1]["timestamp"]-old_dataset[0]["timestamp"])/(old_dataset[-1]["episode_index"]+1)
    fps = 31.25
    print(fps)
    features = {'task_index': {'dtype': 'int64', 'shape': (1,), 'names': None}}
    for feature in old_dataset.features:
        if feature in ["index", "episode_index"]:
            continue
        elif isinstance(old_dataset.features[feature], datasets.features.image.Image):
            features[feature] = {'dtype': 'video',
                                 'shape': np.array(old_dataset[0][feature]).shape,
                                 'names': ['height', 'width', 'channels'],
                                 'info': None}
        elif isinstance(old_dataset.features[feature], datasets.features.features.Sequence):
            features[feature] = {'dtype': old_dataset.features[feature].feature.dtype,
                                 'shape': (old_dataset.features[feature].length,)}
        elif isinstance(old_dataset.features[feature], datasets.features.features.Value):
            features[feature] = {'dtype': old_dataset.features[feature].dtype,
                                 'shape': (1,)}
    print(features)
    new_dataset = LeRobotDataset.create(
        "eufrizz/" + dataset_name,
        fps,
        root=args.out_dir + dataset_name,
        features=features,
        use_videos=True,
        image_writer_processes=0,
        image_writer_threads=2,
    )

    ep_idx = 0
    for frame in tqdm(old_dataset):
        if frame["episode_index"] != ep_idx:
            new_dataset.save_episode(args.description)
            ep_idx += 1
            print(f"Episode {ep_idx}")
        frame.pop('index')
        frame.pop('episode_index')
        frame.pop('frame_index')
        frame["timestamp"] -= old_dataset[0]["timestamp"]
        new_dataset.add_frame(frame)
    
    new_dataset.save_episode(args.description)
    