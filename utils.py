import numpy as np
import torch
import os
import h5py
import glob
from torch.utils.data import TensorDataset, DataLoader, Dataset

import IPython
e = IPython.embed

class EpisodicDataset(torch.utils.data.Dataset):
    def __init__(self, episode_ids, dataset_dir, camera_names, norm_stats):
        super(EpisodicDataset).__init__()
        self.episode_ids = episode_ids
        self.dataset_dir = dataset_dir
        self.camera_names = camera_names
        self.norm_stats = norm_stats
        self.is_sim = None
        self.__getitem__(0) # initialize self.is_sim

    def __len__(self):
        return len(self.episode_ids)

    def __getitem__(self, index):
        sample_full_episode = False # hardcode

        episode_id = self.episode_ids[index]
        dataset_path = os.path.join(self.dataset_dir, f'episode_{episode_id}.hdf5')
        with h5py.File(dataset_path, 'r') as root:
            is_sim = root.attrs['sim']
            original_action_shape = root['/action'].shape
            episode_len = original_action_shape[0]
            if sample_full_episode:
                start_ts = 0
            else:
                start_ts = np.random.choice(episode_len)
            qpos = root['/observations/qpos'][start_ts]
            qvel = root['/observations/qvel'][start_ts]
            image_dict = dict()
            for cam_name in self.camera_names:
                image_dict[cam_name] = root[f'/observations/images/{cam_name}'][start_ts]
            if is_sim:
                action = root['/action'][start_ts:]
                action_len = episode_len - start_ts
            else:
                action = root['/action'][max(0, start_ts - 1):]
                action_len = episode_len - max(0, start_ts - 1)

        self.is_sim = is_sim
        padded_action = np.zeros(original_action_shape, dtype=np.float32)
        padded_action[:action_len] = action
        is_pad = np.zeros(episode_len)
        is_pad[action_len:] = 1

        all_cam_images = []
        for cam_name in self.camera_names:
            all_cam_images.append(image_dict[cam_name])
        all_cam_images = np.stack(all_cam_images, axis=0)

        image_data = torch.from_numpy(all_cam_images)
        qpos_data = torch.from_numpy(qpos).float()
        action_data = torch.from_numpy(padded_action).float()
        is_pad = torch.from_numpy(is_pad).bool()

        image_data = torch.einsum('k h w c -> k c h w', image_data)
        image_data = image_data / 255.0
        action_data = (action_data - self.norm_stats["action_mean"]) / self.norm_stats["action_std"]
        qpos_data = (qpos_data - self.norm_stats["qpos_mean"]) / self.norm_stats["qpos_std"]

        return image_data, qpos_data, action_data, is_pad


def get_norm_stats(dataset_dir, num_episodes):
    all_qpos_data = []
    all_action_data = []
    for episode_idx in range(num_episodes):
        dataset_path = os.path.join(dataset_dir, f'episode_{episode_idx}.hdf5')
        with h5py.File(dataset_path, 'r') as root:
            qpos = root['/observations/qpos'][()]
            qvel = root['/observations/qvel'][()]
            action = root['/action'][()]
        all_qpos_data.append(torch.from_numpy(qpos))
        all_action_data.append(torch.from_numpy(action))
    all_qpos_data = torch.stack(all_qpos_data)
    all_action_data = torch.stack(all_action_data)

    action_mean = all_action_data.mean(dim=[0, 1], keepdim=True)
    action_std = all_action_data.std(dim=[0, 1], keepdim=True)
    action_std = torch.clip(action_std, 1e-2, np.inf)

    qpos_mean = all_qpos_data.mean(dim=[0, 1], keepdim=True)
    qpos_std = all_qpos_data.std(dim=[0, 1], keepdim=True)
    qpos_std = torch.clip(qpos_std, 1e-2, np.inf)

    stats = {"action_mean": action_mean.numpy().squeeze(), "action_std": action_std.numpy().squeeze(),
             "qpos_mean": qpos_mean.numpy().squeeze(), "qpos_std": qpos_std.numpy().squeeze(),
             "example_qpos": qpos}
    return stats


def load_data(dataset_dir, num_episodes, camera_names, batch_size_train, batch_size_val):
    print(f'\nData from: {dataset_dir}\n')
    train_ratio = 0.8
    shuffled_indices = np.random.permutation(num_episodes)
    train_indices = shuffled_indices[:int(train_ratio * num_episodes)]
    val_indices = shuffled_indices[int(train_ratio * num_episodes):]
    norm_stats = get_norm_stats(dataset_dir, num_episodes)
    train_dataset = EpisodicDataset(train_indices, dataset_dir, camera_names, norm_stats)
    val_dataset = EpisodicDataset(val_indices, dataset_dir, camera_names, norm_stats)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True, pin_memory=True, num_workers=1, prefetch_factor=1)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size_val, shuffle=True, pin_memory=True, num_workers=1, prefetch_factor=1)
    return train_dataloader, val_dataloader, norm_stats, train_dataset.is_sim


### env utils

def sample_box_pose():
    x_range = [0.0, 0.2]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]
    ranges = np.vstack([x_range, y_range, z_range])
    cube_position = np.random.uniform(ranges[:, 0], ranges[:, 1])
    cube_quat = np.array([1, 0, 0, 0])
    return np.concatenate([cube_position, cube_quat])

def sample_insertion_pose():
    x_range = [0.1, 0.2]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]
    ranges = np.vstack([x_range, y_range, z_range])
    peg_position = np.random.uniform(ranges[:, 0], ranges[:, 1])
    peg_quat = np.array([1, 0, 0, 0])
    peg_pose = np.concatenate([peg_position, peg_quat])
    x_range = [-0.2, -0.1]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]
    ranges = np.vstack([x_range, y_range, z_range])
    socket_position = np.random.uniform(ranges[:, 0], ranges[:, 1])
    socket_quat = np.array([1, 0, 0, 0])
    socket_pose = np.concatenate([socket_position, socket_quat])
    return peg_pose, socket_pose

### helper functions

def compute_dict_mean(epoch_dicts):
    result = {k: None for k in epoch_dicts[0]}
    num_items = len(epoch_dicts)
    for k in result:
        value_sum = 0
        for epoch_dict in epoch_dicts:
            value_sum += epoch_dict[k]
        result[k] = value_sum / num_items
    return result

def detach_dict(d):
    new_d = dict()
    for k, v in d.items():
        new_d[k] = v.detach()
    return new_d

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)


###############################################################################
# LIBERO dataset utilities
###############################################################################

class LIBERODataset(Dataset):
    """
    Original LIBERO dataset — returns continuous actions.
    """
    def __init__(self, dataset_path, camera_names, norm_stats):
        super().__init__()
        self.camera_names = camera_names
        self.norm_stats = norm_stats
        self.samples = []
        if os.path.isdir(dataset_path):
            hdf5_files = sorted(glob.glob(os.path.join(dataset_path, '*.hdf5')))
        else:
            hdf5_files = [dataset_path]
        for hdf5_path in hdf5_files:
            with h5py.File(hdf5_path, 'r') as f:
                demos = sorted(f['data'].keys())
                for demo_key in demos:
                    ep_len = f[f'data/{demo_key}/actions'].shape[0]
                    self.samples.append((hdf5_path, demo_key, ep_len))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        hdf5_path, demo_key, ep_len = self.samples[index]
        start_ts = np.random.choice(ep_len)
        with h5py.File(hdf5_path, 'r') as root:
            demo = root[f'data/{demo_key}']
            joint_states = demo['obs/joint_states'][start_ts]
            gripper_states = demo['obs/gripper_states'][start_ts]
            qpos = np.concatenate([joint_states, gripper_states])
            action_len = ep_len - start_ts
            action = demo['actions'][start_ts:]
            action_chunk_size = self.norm_stats.get('action_chunk_size', 100)
            padded_action = np.zeros((action_chunk_size, 7), dtype=np.float32)
            is_pad = np.ones(action_chunk_size, dtype=bool)
            actual_len = min(action_len, action_chunk_size)
            padded_action[:actual_len] = action[:actual_len]
            is_pad[:actual_len] = False
            all_cam_images = []
            for cam_name in self.camera_names:
                image = demo[f'obs/{cam_name}'][start_ts]
                image = np.moveaxis(image, -1, 0)
                all_cam_images.append(image)
            all_cam_images = np.stack(all_cam_images, axis=0)
            qpos = (qpos - self.norm_stats['qpos_mean']) / (self.norm_stats['qpos_std'] + 1e-6)
            padded_action = (padded_action - self.norm_stats['action_mean']) / (self.norm_stats['action_std'] + 1e-6)
            all_cam_images = all_cam_images / 255.0
        qpos = torch.from_numpy(qpos).float()
        padded_action = torch.from_numpy(padded_action).float()
        is_pad = torch.from_numpy(is_pad).bool()
        all_cam_images = torch.from_numpy(all_cam_images).float()
        return all_cam_images, qpos, padded_action, is_pad


class LIBEROTokenizedDataset(Dataset):
    """
    LIBERO dataset that returns tokenized action sequences.
    Works with any ActionTokenizer subclass (FAST, VQ-VAE, etc.).

    Returns:
        images: (num_cams, 3, H, W)
        qpos: (state_dim,) — normalized
        action_tokens: (max_token_len,) LongTensor — padded FAST token IDs
        is_pad: (max_token_len,) bool — True at pad positions
    """
    def __init__(self, dataset_path, camera_names, norm_stats, fast_wrapper):
        super().__init__()
        self.camera_names = camera_names
        self.norm_stats = norm_stats
        self.fast_wrapper = fast_wrapper
        self.samples = []
        if os.path.isdir(dataset_path):
            hdf5_files = sorted(glob.glob(os.path.join(dataset_path, '*.hdf5')))
        else:
            hdf5_files = [dataset_path]
        for hdf5_path in hdf5_files:
            with h5py.File(hdf5_path, 'r') as f:
                demos = sorted(f['data'].keys())
                for demo_key in demos:
                    ep_len = f[f'data/{demo_key}/actions'].shape[0]
                    self.samples.append((hdf5_path, demo_key, ep_len))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        hdf5_path, demo_key, ep_len = self.samples[index]
        start_ts = np.random.choice(ep_len)
        chunk_size = self.fast_wrapper.chunk_size

        with h5py.File(hdf5_path, 'r') as root:
            demo = root[f'data/{demo_key}']

            # Proprioception
            joint_states = demo['obs/joint_states'][start_ts]
            gripper_states = demo['obs/gripper_states'][start_ts]
            qpos = np.concatenate([joint_states, gripper_states])

            # Raw actions (NOT normalized — FAST handles its own normalization)
            action_len = ep_len - start_ts
            action = demo['actions'][start_ts:]
            # Pad/truncate to chunk_size
            action_chunk = np.zeros((chunk_size, 7), dtype=np.float32)
            actual_len = min(action_len, chunk_size)
            action_chunk[:actual_len] = action[:actual_len]

            # Images
            all_cam_images = []
            for cam_name in self.camera_names:
                image = demo[f'obs/{cam_name}'][start_ts]
                image = np.moveaxis(image, -1, 0)
                all_cam_images.append(image)
            all_cam_images = np.stack(all_cam_images, axis=0)

            # Normalize qpos (but NOT actions — FAST does its own)
            qpos = (qpos - self.norm_stats['qpos_mean']) / (self.norm_stats['qpos_std'] + 1e-6)
            all_cam_images = all_cam_images / 255.0

        # Tokenize the action chunk with FAST
        tokens, token_len = self.fast_wrapper.encode(action_chunk)  # (max_token_len,), scalar

        # Build is_pad mask for tokens
        max_token_len = self.fast_wrapper.max_token_len
        is_pad = torch.ones(max_token_len, dtype=torch.bool)
        is_pad[:token_len] = False

        qpos = torch.from_numpy(qpos).float()
        all_cam_images = torch.from_numpy(all_cam_images).float()

        return all_cam_images, qpos, tokens, is_pad


def get_libero_norm_stats(dataset_path, camera_names):
    """Compute normalization statistics across all demos."""
    all_qpos = []
    all_actions = []
    if os.path.isdir(dataset_path):
        hdf5_files = sorted(glob.glob(os.path.join(dataset_path, '*.hdf5')))
    else:
        hdf5_files = [dataset_path]
    for hdf5_path in hdf5_files:
        with h5py.File(hdf5_path, 'r') as f:
            for demo_key in sorted(f['data'].keys()):
                demo = f[f'data/{demo_key}']
                joint_states = demo['obs/joint_states'][()]
                gripper_states = demo['obs/gripper_states'][()]
                qpos = np.concatenate([joint_states, gripper_states], axis=-1)
                actions = demo['actions'][()]
                all_qpos.append(qpos)
                all_actions.append(actions)
    all_qpos = np.concatenate(all_qpos, axis=0)
    all_actions = np.concatenate(all_actions, axis=0)
    qpos_std = np.clip(all_qpos.std(axis=0), 1e-2, np.inf)
    action_std = np.clip(all_actions.std(axis=0), 1e-2, np.inf)
    stats = {
        'qpos_mean': all_qpos.mean(axis=0).astype(np.float32),
        'qpos_std': qpos_std.astype(np.float32),
        'action_mean': all_actions.mean(axis=0).astype(np.float32),
        'action_std': action_std.astype(np.float32),
    }
    return stats


def load_libero_data(dataset_path, camera_names, batch_size, chunk_size):
    """Load LIBERO data with continuous actions (original mode)."""
    print(f'\nLIBERO data from: {dataset_path}\n')
    stats = get_libero_norm_stats(dataset_path, camera_names)
    stats['action_chunk_size'] = chunk_size
    full_dataset = LIBERODataset(dataset_path, camera_names, stats)
    print(f'Total LIBERO samples (demos): {len(full_dataset)}')
    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    return train_loader, val_loader, stats, full_dataset


def load_libero_data_tokenized(dataset_path, camera_names, batch_size, fast_wrapper):
    """Load LIBERO data with FAST-tokenized actions."""
    print(f'\nLIBERO FAST-tokenized data from: {dataset_path}\n')
    stats = get_libero_norm_stats(dataset_path, camera_names)
    full_dataset = LIBEROTokenizedDataset(dataset_path, camera_names, stats, fast_wrapper)
    print(f'Total LIBERO tokenized samples (demos): {len(full_dataset)}')
    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    return train_loader, val_loader, stats, full_dataset