"""Canonical-task sampling from an accepted episode/split manifest."""
from collections import Counter

TASKS = ("fill_pen_holder", "classify_objects", "put_bottles_into_dustbin",
         "play_tic_tac_toe", "fill_egg_holder", "organize_table", "make_kong",
         "play_stacking_toy")


def frame_weights(episode_ids, episodes, setting):
    if setting not in {'ST-Organize', 'MT-Uniform', 'MT-Organize50'}:
        raise ValueError('Unknown baseline setting')
    desired = {"organize_table": 1.0} if setting == "ST-Organize" else {
        task: (.5 if task == "organize_table" else .5 / 7)
        if setting == "MT-Organize50" else 1 / 8 for task in TASKS}
    counts = Counter()
    labels = []
    for index in episode_ids:
        entry = episodes[str(int(index))]
        if entry["split"] not in {"train", "val", "test"} or entry["task"] not in TASKS:
            raise ValueError("Invalid episode task or split")
        task = entry["task"] if entry["split"] == "train" else None
        labels.append(task)
        if task in desired:
            counts[task] += 1
    if set(counts) != set(desired):
        raise ValueError("Every required task needs training frames")
    return [desired[task] / counts[task] if task in desired else 0.0 for task in labels]


def dataset_sources(reports):
    sources = []
    episode_offset = frame_offset = 0
    for task in TASKS:
        report = reports[task]
        sources.append({"task": task, "repo_id": report["repo_id"], "root": report["dataset_root"],
                        "episode_offset": episode_offset, "frame_offset": frame_offset,
                        "episodes": report["total_episodes"], "frames": report["total_frames"]})
        episode_offset += report["total_episodes"]
        frame_offset += report["total_frames"]
    return sources


def prompt_from_task(item):
    return {**item, "prompt": item["task"]}


def install(data_module, manifest, repo_id, setting, seed):
    import torch

    from pathlib import Path
    from lerobot.datasets import lerobot_dataset
    from native_multi_dataset import NumericQueryLeRobotDataset

    lerobot_dataset.LeRobotDataset = NumericQueryLeRobotDataset

    original = data_module.TorchDataLoader
    entry = manifest["datasets"][repo_id]
    episodes = entry["episodes"]
    sources = entry.get("source_datasets")
    if sources:
        original_create = data_module.create_torch_dataset

        def create(data_config, action_horizon, model_config):
            if data_config.repo_id != repo_id:
                return original_create(data_config, action_horizon, model_config)
            dataset = lerobot_dataset.MultiLeRobotDataset(
                [source["repo_id"] for source in sources], root=Path(sources[0]["root"]).parent,
                delta_timestamps={key: [i / entry["fps"] for i in range(action_horizon)]
                                  for key in data_config.action_sequence_keys},
                video_backend=data_config.video_backend)
            return (data_module.TransformedDataset(dataset, [prompt_from_task])
                    if data_config.prompt_from_task else dataset)

        data_module.create_torch_dataset = create

    def loader(dataset, **kwargs):
        raw = dataset
        while not hasattr(raw, "meta") and not isinstance(raw, lerobot_dataset.MultiLeRobotDataset):
            raw = raw._dataset
        if sources:
            ids = [int(ep) + source["episode_offset"]
                   for source, part in zip(sources, raw._datasets, strict=True)
                   for ep in part.hf_dataset.with_format(None)["episode_index"]]
        else:
            raw[0]  # LeRobot 0.4 lazily loads hf_dataset.
            ids = raw.hf_dataset.with_format(None)["episode_index"]
        weights = frame_weights(ids, episodes, setting)
        n_train = sum(weight > 0 for weight in weights)
        if n_train < kwargs["local_batch_size"]:
            raise ValueError("Fewer training frames than the effective batch")
        generator = torch.Generator().manual_seed(seed)
        kwargs["sampler"] = torch.utils.data.WeightedRandomSampler(
            weights, n_train, replacement=True, generator=generator)
        kwargs["shuffle"] = False
        return original(dataset, **kwargs)

    data_module.TorchDataLoader = loader


def self_check():
    import math
    episodes = {str(i): {"task": task, "split": "train"} for i, task in enumerate(TASKS)}
    episodes['8'] = {"task": "organize_table", "split": "test"}
    episodes['9'] = {"task": "organize_table", "split": "val"}
    ids = [index for index in range(8) for _ in range(index + 1)] + [8, 9]
    for setting in ['ST-Organize', 'MT-Uniform', 'MT-Organize50']:
        weights = frame_weights(ids, episodes, setting)
        assert weights[-2:] == [0., 0.]
        assert math.isclose(sum(weights), 1.)
        mass = {task: sum(w for i, w in zip(ids, weights) if episodes[str(i)]['task'] == task)
                for task in TASKS}
        for task, actual in mass.items():
            expected = (float(task == 'organize_table') if setting == 'ST-Organize' else
                        (.5 if task == 'organize_table' else .5/7) if setting == 'MT-Organize50'
                        else 1/8)
            assert math.isclose(actual, expected, abs_tol=1e-12), (setting, task, actual)
    try:
        frame_weights([0], episodes, 'MT-Uniform')
    except ValueError:
        pass
    else:
        raise AssertionError('Missing tasks were accepted')
    print('TASK_SAMPLING_CHECK_PASSED: task mass, unequal episode lengths, zero held-out mass')


if __name__ == '__main__':
    self_check()
