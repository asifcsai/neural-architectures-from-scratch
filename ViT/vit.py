import torch
from torch import nn
from transformer_layer import TransformerLayer
from tools import get_patches, positional_encoding





class ViT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
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
        
        self.patch_embedding_layer = nn.Linear(self.patch_height * self.patch_width * 3, self.d_model)


        # Shape of class_token : (1, 1, d_model)
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.d_model))

        # Instead of calculating sinusoidal embeddings and concatenating pos_cls:
        # Shape of pos_embedding :(1, num_patches + 1, d_model)
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches + 1, self.d_model))



        self.layers = nn.ModuleList([
            TransformerLayer(self.d_model, self.num_heads, self.mlp_ratio, self.dropout) for _ in range(self.num_layers)    
        ])

        self.out_norm = nn.LayerNorm(self.d_model)
        self.output_layer = nn.Linear(self.d_model, 1000) # Assuming 1000 classes for classification
   


    def forward(self, x):
        # Shape of x : (B, C, H, W)
        # Shape of patch_embeddings : (B, N, self.patch_height * self.patch_width * C)
        x = get_patches(x, self.config)
        # Shape of patch_embeddings : (B, N, self.patch_height * self.patch_width * C) -> (B, N, d_model)
        x = self.patch_embedding_layer(x)


        # Shape of class_token : (1, 1, d_model) -> (B, 1, d_model)
        class_tokens = self.cls_token.expand(x.size(0), -1, -1) # (B, 1, d_model)
        # Shape of x : (B, N, d_model) -> (B, N+1, d_model)
        x = torch.cat([class_tokens, x], dim=1) # (B, N+1, d_model)
        
        # Shape of pos_embedding : (1, N+1, d_model)
        # Shape of x : (B, N+1, d_model) -> (B, N+1, d_model)
        x = x + self.pos_embedding # (B, N+1, d_model)

        for layer in self.layers:
            x = layer(x)


        cls_output = x[:, 0] # (B, d_model)
        cls_output = self.out_norm(cls_output)
        output = self.output_layer(cls_output) # (B, num_classes)
        return output





import yaml




if __name__ == "__main__":

    config = yaml.safe_load(open('config.yaml'))
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    config['device'] = device
    print(config)

    # 2. Instantiate the model
    model = ViT(
        config= config
    )

    # 3. Create dummy inputs (Images and Timesteps)
    dummy_x = torch.randn(config['batch_size'], 3, config['img_height'], config['img_width']).to(device)
  

    # 4. Forward pass
    output = model(dummy_x)

    print(f"Input shape:  {dummy_x.shape}")
    print(f"Output shape: {output.shape}") 
    # Output will perfectly match Input shape!










# Image
#  ↓
# Split into patches
#  ↓
# Linear projection (patch embedding)
#  ↓
# Add positional encoding
#  ↓
# Add [CLS] token
#  ↓
# Transformer encoder layers
#  ↓
# Take CLS token output
#  ↓
# LayerNorm
#  ↓
# MLP classifier
#  ↓
# Class prediction


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

        






