import torch
from torch import nn
from transformer_layer import TransformerLayer
from tools import get_patches, positional_encoding , get_time_embedding





class DiT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.device = config['device']
        self.img_height = config['img_height']
        self.img_width = config['img_width']
        self.patch_height = config['patch_height']
        self.patch_width = config['patch_width']
        self.num_patches = (self.img_height // self.patch_height) * (self.img_width // self.patch_width)
        self.d_model = config['d_model']
        self.num_heads = config['num_heads']
        self.num_layers = config['num_layers']
        self.mlp_ratio = config['mlp_ratio']
        self.dropout = config['dropout']  
        self.time_emb_dim = config['time_emb_dim']  
        
        self.patch_embedding_layer = nn.Linear(self.patch_height * self.patch_width * 3, self.d_model)
        self.time_embedding_layer = nn.Sequential(
            nn.Linear(self.time_emb_dim, self.d_model),
            nn.GELU(),
            nn.Linear(self.d_model, self.d_model)
        )



        self.layers = nn.ModuleList([
            TransformerLayer(self.d_model, self.num_heads, self.mlp_ratio, self.dropout) for _ in range(self.num_layers)    
        ])

        self.adaptive_layer = nn.Sequential(
            nn.GELU(),
            nn.Linear(self.d_model, self.d_model * 2)
        )

        self.out_norm = nn.LayerNorm(self.d_model, eps=1e-6, elementwise_affine=True)
        # Output is the predicted noise for each patch, which has the same shape as the input patch.  
        self.output_layer = nn.Linear(self.d_model, self.patch_height * self.patch_width * 3) 

        # DiT Initialization
        torch.nn.init.normal_(self.time_embedding_layer[0].weight, std=0.2)
        torch.nn.init.normal_(self.time_embedding_layer[-1].weight, std=0.2)

        torch.nn.init.constant_(self.adaptive_layer[-1].weight, 0)
        torch.nn.init.constant_(self.adaptive_layer[-1].bias, 0)

        torch.nn.init.constant_(self.output_layer.weight, 0)
        torch.nn.init.constant_(self.output_layer.bias, 0)
 
   


    def forward(self, x,t):
        B, C, H, W = x.shape
        assert H == self.img_height and W == self.img_width, f'Input image size must be ({self.img_height}, {self.img_width})'  
        # Shape of x : (B, C, H, W)
        # Shape of t : (B,)
        # Shape of patch_embeddings : (B, N, self.patch_height * self.patch_width * C)
        x = get_patches(x, self.config)
        # Shape of patch_embeddings : (B, N, self.patch_height * self.patch_width * C) -> (B, N, d_model)
        x = self.patch_embedding_layer(x)


        # Shape of time_embedding : (B,) -> (B, time_emb_dim)
        time_embedding = get_time_embedding(t, self.config).unsqueeze(1) # (B, 1, time_emb_dim)
        # Shape of time_embedding : (B, 1, time_emb_dim) -> (B, 1, d_model)
        time_embedding = self.time_embedding_layer(time_embedding)

        # Shape of pos_embedding: (1, N, d_model)  :  N = num_patches
        pos_embedding = positional_encoding(self.config, self.device)  

        # Shape of adaptive_params :(B, 1, d_model) -> (B, 1, 2*d_model)
        adaptive_params = self.adaptive_layer(time_embedding) # (B, 1, 2*d_model)

        # Each of shape (B, 1, d_model)
        pre_out_shift, pre_out_scale = adaptive_params.chunk(2, dim=-1) 

        x = x + pos_embedding

        # Shape of x : (B, N, d_model)
        # Shape of time_embedding : (B, 1, d_model)
        for layer in self.layers:
            # Shape of x : (B, N, d_model) -> (B, N, d_model)
            x = layer(x, time_embedding)

        # Shape of x : (B, N, d_model) -> (B, N, d_model)
        x = self.out_norm(x)*(pre_out_scale + 1) + pre_out_shift # (B, N, d_model)
        # Shape of output : (B, N, patch_height * patch_width * 3)
        output = self.output_layer(x)
        output = output.view(B, self.img_height // self.patch_height, self.img_width // self.patch_width, self.patch_height, self.patch_width, 3)
        # Shape of output : (B, 3, num_patches_h, patch_height, num_patches_w, patch_width)
        output = output.permute(0, 5, 1, 3, 2, 4).contiguous() 

        output = output.view(B, 3, self.img_height, self.img_width)

        return output








import yaml




if __name__ == "__main__":

    config = yaml.safe_load(open('config.yaml'))
    print(config)
    # 1. Define hyperparameters
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    config['device'] = device

    # 2. Instantiate the model
    model = DiT(
        config= config
    )

    # 3. Create dummy inputs (Images and Timesteps)
    dummy_x = torch.randn(config['batch_size'], 3, config['img_height'], config['img_width']).to(device)
    dummy_t = torch.randint(0, 1000, (config['batch_size'],)).to(device) # Random timesteps

    # 4. Forward pass
    output = model(dummy_x, dummy_t)

    print(f"Input shape:  {dummy_x.shape}")
    print(f"Output shape: {output.shape}") 
    # Output will perfectly match Input shape!





# Noisy Image / Spatial Latent
#  ↓
# Split into patches
#  ↓
# Linear projection (patch embedding)
#  ↓
# Add positional encoding (Your 2D Sinusoidal embeddings)
#  ↓
#    <-- (Timestep 't' and Class 'y' embeddings are processed separately here)
#  ↓
# DiT Transformer blocks (Standard Self-Attention, but MLP and LayerNorm 
#                         are modulated by 't' and 'y' via adaLN)
#  ↓
# Final LayerNorm (also modulated by adaLN) applied to ALL patch tokens
#  ↓
# Linear projection (Expand channels to match original patch dimensions)
#  ↓
# Unpatchify (Reshape the sequence back into a 2D spatial grid)
#  ↓
# Output prediction (Usually the predicted noise or un-noised latent)



# The Standard Practice:
# In PyTorch, standard ViT implementations use a 2D Convolution with the kernel_size and stride set to the patch_size. 
# This perfectly mimics extracting non-overlapping patches and applying a linear projection simultaneously, 
# and it is highly optimized at the CUDA level.

# Standard Implementation:
# You can completely delete get_patches and replace your patch_embedding_layer with this:
# # In ViT.__init__
# self.patch_embedding_layer = nn.Conv2d(
#     in_channels=3, 
#     out_channels=self.d_model, 
#     kernel_size=(self.patch_height, self.patch_width), 
#     stride=(self.patch_height, self.patch_width)
# )

# # In ViT.forward
# x = self.patch_embedding_layer(x) # Shape: (B, d_model, H_patches, W_patches)
# x = x.flatten(2).transpose(1, 2)  # Shape: (B, num_patches, d_model)

        






