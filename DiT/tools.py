import torch
from torch import nn


def get_patches(x, config):
    B, C, H, W = x.shape
    patch_height = config['patch_height']
    patch_width = config['patch_width']
    num_patches_h = H // patch_height
    num_patches_w = W // patch_width
    num_patches = num_patches_h * num_patches_w
    assert H % patch_height == 0 and W % patch_width == 0, 'Image dimensions must be divisible by patch size'

    
    # Shape of x : (B, C, H, W) -> (B, C, num_patches_h, patch_height, num_patches_w, patch_width)
    x = x.view(B, C, num_patches_h, patch_height, num_patches_w, patch_width)
    # Shape of x : (B, C, num_patches_h, patch_height, num_patches_w, patch_width) -> (B, num_patches_h, num_patches_w, patch_height, patch_width, C)
    x = x.permute(0, 2, 4, 3, 5, 1).contiguous() 
    # Shape of x : (B, num_patches_h, num_patches_w, patch_height, patch_width, C) -> (B, num_patches, patch_height * patch_width * C)
    x = x.view(B, num_patches, patch_height * patch_width * C)
    return x



def get_time_embedding(t, config):
    time_emb_dim = config['time_emb_dim']
    device = t.device
    # Shape of t: (B,) -> (B, 1)
    t = t[:, None]
    
    # factor = 10000^(2i/time_emb_dim)
    # Shape of factor : (time_emb_dim // 2,)
    factor = 10000 ** (torch.arange(
        start=0,
        end=time_emb_dim // 2,
        device=device,
        dtype=torch.float32
    )/(time_emb_dim // 2))


    # Shape of t: (B, 1) -> (B, time_emb_dim // 2)
    t = t.repeat(1, time_emb_dim // 2)
    # Shape of t: (B, time_emb_dim // 2) -> (B, time_emb_dim // 2)
    # here div will be in last dim and rest of dim will be broadcasted. 
    # i.e. data like [[0, 0, 0, ...], [0, 0, 0, ...], [0, 0, 0, ...], ...] -> [[0, 0, 0, ...], [0, 0, 0, ...], [0, 0, 0, ...], ...]
    t = t/factor
    # Shape of t: (B, time_emb_dim // 2) -> (B, time_emb_dim)
    t = torch.cat([torch.sin(t), torch.cos(t)], dim=-1)
    return t





def positional_encoding(config, device):
    patch_height = config['patch_height']
    patch_width = config['patch_width']
    num_patches_h = config['img_height'] // patch_height
    num_patches_w = config['img_width'] // patch_width
    num_patches = num_patches_h * num_patches_w
    d_model = config['d_model']
    
    # Shape of grid_h: (num_patches_h,)
    grid_h = torch.arange(num_patches_h, device=device)
    # Shape of grid_w: (num_patches_w,)
    grid_w = torch.arange(num_patches_w, device=device)
    # Shape of each of h_grid and w_grid: (num_patches_h, num_patches_w)
    # i.e. data like for h_grid  [[0, 0, 0, ...], ... [1, 1, 1, ...], ... [2, 2, 2, ...], ...] 
    # for for w_grid             [[0, 1, 2, ...], ... [0, 1, 2, ...], ... [0, 1, 2, ...], ...]
    grid = torch.meshgrid(grid_h, grid_w , indexing='ij') 

    # Shape of grid: (2, num_patches_h, num_patches_w)
    grid = torch.stack(grid, dim=0) 
    

    grid = grid.view(2, -1) # (2, num_patches)
    horizontal_pos = grid[0]
    vertical_pos = grid[1]

    # factor = 10000^(2i/d_model)
    # Shape of factor : (d_model // 4,)
    factor = 10000 **(torch.arange(
        start=0, 
        end=d_model // 4,
        device=device,
        dtype=torch.float32
    )/(d_model // 4)
    ) 
    
    # Shape of horizontal_pos: (num_patches,) -> (num_patches, 1)
    horizontal_pos = horizontal_pos[:, None]
    # shape of horizontal_pos: (num_patches, 1) -> (num_patches, d_model // 4)
    # i.e. data like [[0], [0], [0]...] -> [[0, 0, 0, ...], [0, 0, 0, ...], [0, 0, 0, ...], ...]
    horizontal_pos = horizontal_pos.repeat(1, d_model // 4)
    # shape of horizontal_pos: (num_patches, d_model // 4) -> (num_patches, d_model // 4)
    # here div will be in last dim and rest of dim will be broadcasted. i.e. data like [[0, 0, 0, ...], [0, 0, 0, ...], [0, 0, 0, ...], ...] -> [[0, 0, 0, ...], [0, 0, 0, ...], [0, 0, 0, ...], ...]
    horizontal_pos = horizontal_pos / factor

    # Shape of vertical_pos: (num_patches,) -> (num_patches, 1)
    vertical_pos = vertical_pos[:, None]
    vertical_pos = vertical_pos.repeat(1, d_model // 4)
    vertical_pos = vertical_pos / factor
    
    # Shape of horizontal_pos_embedding and vertical_pos_embedding: (num_patches, d_model // 2)
    horizontal_pos_embedding = torch.cat([torch.sin(horizontal_pos), torch.cos(horizontal_pos)], dim=-1) # (num_patches, d_model // 2)
    vertical_pos_embedding = torch.cat([torch.sin(vertical_pos), torch.cos(vertical_pos)], dim=-1) # (num_patches, d_model // 2)

    # Shape of pos_embedding: (num_patches, d_model) -> (1, num_patches, d_model)
    pos_embedding = torch.cat([horizontal_pos_embedding, vertical_pos_embedding], dim=-1).unsqueeze(0) # (1, num_patches, d_model)
    # Shape of pos_embedding: (1, num_patches, d_model)
    return pos_embedding




