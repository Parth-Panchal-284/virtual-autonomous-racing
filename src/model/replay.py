import torch
import numpy as np

class ReplayBuffer:
    def __init__(
        self,
        capacity: int,
        action_dim: int,
        device: torch.device,
        img_shape: tuple[int, int, int] | None,
        vec_dim: int | None,
    ):
        self.capacity = capacity
        self.device = device
        self.ptr = 0
        self.size = 0

        self.has_img = img_shape is not None
        self.has_vec = vec_dim is not None and vec_dim > 0

        if self.has_img:
            c, h, w = img_shape  # type: ignore
            self.imgs = np.zeros((capacity, c, h, w), dtype=np.uint8)
            self.next_imgs = np.zeros((capacity, c, h, w), dtype=np.uint8)

        if self.has_vec:
            self.vecs = np.zeros((capacity, vec_dim), dtype=np.float32)  # type: ignore
            self.next_vecs = np.zeros((capacity, vec_dim), dtype=np.float32)  # type: ignore

        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.dones = np.zeros((capacity, 1), dtype=np.float32)

    def add(self, obs_img, obs_vec, action, reward, next_img, next_vec, done):
        i = self.ptr

        if self.has_img:
            self.imgs[i] = obs_img.astype(np.uint8)
            self.next_imgs[i] = next_img.astype(np.uint8)

        if self.has_vec:
            self.vecs[i] = obs_vec.astype(np.float32)
            self.next_vecs[i] = next_vec.astype(np.float32)

        self.actions[i] = action.astype(np.float32)
        self.rewards[i] = float(reward)
        self.dones[i] = float(done)

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int):
        idx = np.random.randint(0, self.size, size=batch_size)

        img_t = None
        next_img_t = None
        vec_t = None
        next_vec_t = None

        if self.has_img:
            img_t = torch.from_numpy(self.imgs[idx].astype(np.float32)).to(self.device)
            next_img_t = torch.from_numpy(self.next_imgs[idx].astype(np.float32)).to(self.device)

        if self.has_vec:
            vec_t = torch.from_numpy(self.vecs[idx]).to(self.device)
            next_vec_t = torch.from_numpy(self.next_vecs[idx]).to(self.device)

        actions_t = torch.from_numpy(self.actions[idx]).to(self.device)
        rewards_t = torch.from_numpy(self.rewards[idx]).to(self.device)
        dones_t = torch.from_numpy(self.dones[idx]).to(self.device)

        return (img_t, vec_t), actions_t, rewards_t, (next_img_t, next_vec_t), dones_t

    def __len__(self):
        return self.size