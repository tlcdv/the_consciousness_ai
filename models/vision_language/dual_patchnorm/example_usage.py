import torch
from models.vision.dual_patchnorm import DualPatchNormConfig, DualPatchNorm

def main():
    """Example usage of DualPatchNorm"""
    
    # Create configuration
    config = DualPatchNormConfig(
        patch_size=(16, 16),
        hidden_size=768,
        eps=1e-6,
        elementwise_affine=True,
        dropout=0.1,
        num_heads=12
    )
    
    # Initialize model
    dual_patchnorm = DualPatchNorm(config)
    
    # Create example input
    batch_size = 4
    img_size = (224, 224)
    x = torch.randn(batch_size, *img_size, 3)
    
    # Forward pass
    output = dual_patchnorm(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Number of patches: {output.shape[1]}")
    print(f"Feature dimension: {output.shape[2]}")

if __name__ == "__main__":
    main()