import os
from datetime import datetime
import pathlib

class CurrentRunFolder:
    def __init__(self, run_folder: str):
        self.run_folder = run_folder
        self._create_run_folders()
    
    def _create_run_folders(self):
        os.makedirs(self.run_folder, exist_ok=True)
    
    def get_file_name(self, filename:str, subfolder:str|None=None):
        if(subfolder is not None):
            os.makedirs(str(pathlib.Path(self.run_folder, subfolder)), exist_ok=True)
            return str(pathlib.Path(self.run_folder, subfolder, filename))
        return str(pathlib.Path(self.run_folder, filename))
    
    def get_folder(self, folder: str):
        os.makedirs(str(pathlib.Path(self.run_folder, folder)), exist_ok=True)
        return str(pathlib.Path(self.run_folder, folder))
    
    def get_date_file_name(self, extension:str, subfolder:str|None=None):
        filename = datetime.now().strftime("%Y-%m-%d-%H_%M_%S") + "." + extension
        return self.get_file_name(filename, subfolder)


import numpy as np
import torch

def split_obs(obs):
    """
    Split a tmrl observation tuple into:
      - img_arrays : list of (1, H, W) or (H, W) numpy arrays  (images)
      - vec_arrays : list of flat numpy arrays                  (speed/gear/rpm etc.)

    Anything with more than 1000 elements is treated as an image.
    """
    if not isinstance(obs, (tuple, list)):
        return [], [np.array(obs, dtype=np.float32).flatten()]

    img_arrays, vec_arrays = [], []
    
    for o in obs:
        # print(o.shape)
        arr = np.array(o, dtype=np.float32)
        if arr.ndim >= 2:
            img_arrays.append(arr)
        else:
            vec_arrays.append(arr.flatten())
    return img_arrays, vec_arrays


def obs_to_tensors(obs, device: torch.device):
    """
    Returns (img_tensor, vec_tensor) on `device`.
      img_tensor : (1, C, H, W)  stacked grayscale frames, or None
      vec_tensor : (1, D)        concatenated vector obs,   or None
    """
    img_arrays, vec_arrays = split_obs(obs)

    img_t = None
    if img_arrays:
        img_t = torch.from_numpy(np.array(img_arrays)).to(device)

    vec_t = None
    if vec_arrays:
        vec   = np.concatenate(vec_arrays)
        vec_t = torch.from_numpy(vec).unsqueeze(0).to(device)      # (1, D)

    return img_t, vec_t


def probe_obs_dims(obs):
    """
    Given a sample observation, return (n_img_channels, img_h, img_w, vec_dim).
    Returns None for streams that are absent.
    """
    img_arrays, vec_arrays = split_obs(obs)

    img_channels = img_h = img_w = None
    if img_arrays:
        img_channels = img_arrays[0].shape[0]
        first = np.array(img_arrays[0], dtype=np.float32)
        print(first.shape)
        img_h, img_w = first.shape[1], first.shape[2]

    vec_dim = None
    if vec_arrays:
        vec_dim = sum(a.flatten().size for a in vec_arrays)

    return img_channels, img_h, img_w, vec_dim


if __name__ == "__main__":
    a = CurrentRunFolder("runs/a")

    print(a.get_file_name("b.txt", "c"))