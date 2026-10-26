#!/usr/bin/env python3
"""One real RoboDojo episode for a pipeline check, using LeRobot 0.4.4.

Uses the pinned XPolicyLab RGB decoder and its next-state action convention.
This pilot dataset has no held-out evaluation split and is not an experiment.
"""
import hashlib
import json
import os
from pathlib import Path

import h5py
from huggingface_hub import HfApi, hf_hub_download
from lerobot.datasets.lerobot_dataset import LeRobotDataset
import numpy as np
from XPolicyLab.utils.process_data import decode_image_bit


def main():
    root = Path(os.environ['WM_ROOT'])
    repo_id = 'RoboDojo-organize_table-pilot-arx_x5-joint'
    destination = Path(os.environ['HF_LEROBOT_HOME']) / repo_id
    manifest = root / 'logs/pilot_data.json'
    if destination.exists():
        if manifest.exists():
            print('PILOT_DATA_READY', manifest.read_text())
            return
        raise FileExistsError(f'Incomplete dataset at {destination}; inspect before retrying.')

    source_repo = 'RoboDojo-Benchmark/RoboDojo'
    revision = HfApi().dataset_info(source_repo).sha
    filename = 'data/RoboDojo/organize_table/arx_x5/data/episode_0000000.hdf5'
    path = hf_hub_download(source_repo, filename, repo_type='dataset', revision=revision,
                           local_dir=root / 'data/raw')
    cameras = {'cam_head': 'cam_high', 'cam_left_wrist': 'cam_left_wrist',
               'cam_right_wrist': 'cam_right_wrist'}
    with h5py.File(path, 'r') as episode:
        state = np.concatenate([
            episode['state/left_arm_joint_states'][:],
            episode['state/left_ee_joint_states'][:].reshape(-1, 1),
            episode['state/right_arm_joint_states'][:],
            episode['state/right_ee_joint_states'][:].reshape(-1, 1),
        ], axis=1).astype(np.float32)
        assert state.ndim == 2 and state.shape[1] == 14 and len(state) > 50, state.shape
        assert np.isfinite(state).all()
        action = np.concatenate([state[1:], state[-1:]], axis=0)
        features = {key: {'dtype': 'float32', 'shape': (14,), 'names': None}
                    for key in ['observation.state', 'action']}
        for source, name in cameras.items():
            images = episode[f'vision/{source}/colors']
            assert len(images) == len(state), (source, len(images), len(state))
            image = decode_image_bit(images[0])
            assert image.ndim == 3 and image.shape[-1] == 3, image.shape
            features[f'observation.images.{name}'] = {
                'dtype': 'image', 'shape': image.shape,
                'names': ['height', 'width', 'channels'],
            }
        dataset = LeRobotDataset.create(repo_id=repo_id, fps=50, root=destination,
                                        robot_type='arx_x5', features=features,
                                        use_videos=False, image_writer_threads=4)
        try:
            for index in range(len(state)):
                frame = {'observation.state': state[index], 'action': action[index],
                         'task': 'Organize the table.'}
                for source, name in cameras.items():
                    frame[f'observation.images.{name}'] = decode_image_bit(
                        episode[f'vision/{source}/colors'][index])
                dataset.add_frame(frame)
            dataset.save_episode()
        finally:
            dataset.finalize()
            dataset.stop_image_writer()
    loaded = LeRobotDataset(repo_id=repo_id, root=destination)
    assert len(loaded) == len(state)
    assert loaded[0]['observation.state'].shape == (14,)
    with open(path, 'rb') as handle:
        source_sha256 = hashlib.file_digest(handle, 'sha256').hexdigest()
    record = {'repo_id': repo_id, 'source_repo': source_repo, 'source_revision': revision,
              'source_file': filename, 'source_sha256': source_sha256,
              'episodes': 1, 'frames': len(state), 'purpose': 'pipeline_check_only',
              'prompt': 'Organize the table.', 'fps': 50,
              'action_convention': 'XPolicyLab next-state targets; final state repeated',
              'decoder': 'pinned XPolicyLab.utils.process_data.decode_image_bit',
              'features': features}
    manifest.write_text(json.dumps(record, indent=2) + '\n')
    print('PILOT_DATA_READY', json.dumps(record))


if __name__ == '__main__':
    main()

