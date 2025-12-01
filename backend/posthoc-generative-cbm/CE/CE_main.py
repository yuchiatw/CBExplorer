"""
Main module for Concept Exploration (CE) image generation.
Provides API functions to dynamically generate images from the CB-AE model.
"""

import os
import sys
import io
import yaml
import torch
import numpy as np
import itertools
from pathlib import Path
from PIL import Image
from torchvision.utils import save_image

# Add project root to path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from models import cbae_stygan2
from CE.CE_utils import get_concept_index


# Global model cache to avoid reloading
_model_cache = {}
_config_cache = {}


def initialize_model(dataset='cub', expt_name='cbae_stygan2', 
                     tensorboard_name='sup_pl_cls8_cbae.pt', device='cuda:0'):
    """
    Initialize and load the CB-AE model. Uses caching to avoid reloading.
    
    Args:
        dataset: Dataset name (e.g., 'cub', 'celebahq')
        expt_name: Experiment name
        tensorboard_name: Tensorboard checkpoint name
        device: Device to load model on
        
    Returns:
        tuple: (model, config, concept_names)
    """
    cache_key = f"{dataset}_{expt_name}"
    
    # Return cached model if available
    if cache_key in _model_cache:
        print(f"Using cached model for {cache_key}")
        return _model_cache[cache_key], _config_cache[cache_key]
    
    print(f"Loading model for {cache_key}...")
    
    # Change to project root directory
    original_cwd = os.getcwd()
    os.chdir(project_root)
    
    try:
        # Load configuration
        config_file = f"./config/{expt_name}/{dataset}.yaml"
        with open(config_file, 'r') as stream:
            config = yaml.safe_load(stream)
        print(f"Loaded configuration file {config_file}")
        
        # Validate configuration
        if not all([type == 'bin' for type in config['model']['concepts']['types'][:-1]]):
            raise ValueError("All but the last concept must be binary")
        
        # Extract concept information
        concept_names = config['model']['concepts']['concept_names'][:-1]
        print(f"Concept names: {concept_names}")
        
        # Set up model
        model = cbae_stygan2.cbAE_StyGAN2(config)
        
        # Load checkpoint
        cbae_ckpt_path = f'models/checkpoints/{dataset}_{expt_name}.pt'
        model.cbae.load_state_dict(torch.load(cbae_ckpt_path, map_location='cpu'))
        
        # Move to device and set to evaluation mode
        model.to(device)
        model.eval()
        
        print(f"Model loaded successfully on {device}")
        
        # Cache the model and config
        _model_cache[cache_key] = model
        _config_cache[cache_key] = (config, concept_names)
        
        return model, (config, concept_names)
        
    finally:
        # Restore original working directory
        os.chdir(original_cwd)


def manipulate_concepts(model, concepts, combination):
    """
    Manipulate concepts to match the desired binary combination.
    
    Args:
        model: The CB-AE model
        concepts: Original concept tensor
        combination: Tuple of binary values (e.g., (0, 1, 1, 0, 1, 0, 1, 1))
        
    Returns:
        Modified concept tensor
    """
    new_concepts = concepts.clone()
    
    # Set each concept to the desired value in this combination
    for concept_idx, concept_val in enumerate(combination):
        start, end = get_concept_index(model, concept_idx)
        c_concepts = concepts[:, start:end]
        _, num_c = c_concepts.shape
        
        # Swap the max value to the concept we need
        new_c_concepts = c_concepts.clone()
        old_vals = new_c_concepts[:, concept_val].clone()
        max_val, max_ind = torch.max(new_c_concepts, dim=1)
        new_c_concepts[:, concept_val] = max_val
        for swap_idx, (curr_ind, curr_old_val) in enumerate(zip(max_ind, old_vals)):
            new_c_concepts[swap_idx, curr_ind] = curr_old_val
        
        new_concepts[:, start:end] = new_c_concepts
    
    return new_concepts


def generate_image_from_combination(seed, combination, dataset='cub', 
                                    expt_name='cbae_stygan2',
                                    tensorboard_name='sup_pl_cls8_cbae.pt',
                                    device='cuda:0',
                                    return_concepts=False):
    """
    Generate an image for a specific seed and concept combination.
    
    Args:
        seed: Random seed for latent generation
        combination: Binary combination string (e.g., "01101011") or tuple (0,1,1,0,1,0,1,1)
        dataset: Dataset name
        expt_name: Experiment name
        tensorboard_name: Tensorboard checkpoint name
        device: Device to use
        return_concepts: If True, return (image, concept_values) tuple
        
    Returns:
        PIL Image object, or (PIL Image, concept_values list) if return_concepts=True
    """
    # Initialize model (uses cache if already loaded)
    model, (config, concept_names) = initialize_model(dataset, expt_name, 
                                                       tensorboard_name, device)
    
    # Convert string combination to tuple if needed
    if isinstance(combination, str):
        combination = tuple(int(c) for c in combination)
    
    # Set random seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Generate latent vector from seed
    z = torch.randn((1, model.gen.z_dim), device=device)
    latent = model.gen.mapping(z, None, truncation_psi=1.0, truncation_cutoff=None)
    
    # Get original concepts
    concepts = model.cbae.enc(latent)
    
    # Manipulate concepts to match desired combination
    new_concepts = manipulate_concepts(model, concepts, combination)
    
    # Extract concept values (argmax for each concept)
    if return_concepts:
        concept_values = []
        num_concepts = len(combination)
        for concept_idx in range(num_concepts):
            start, end = get_concept_index(model, concept_idx)
            concept_logits = new_concepts[:, start:end]
            argmax_val = torch.argmax(concept_logits, dim=1).item()
            concept_values.append(int(argmax_val))
    
    # Decode and generate image
    new_latent = model.cbae.dec(new_concepts)
    gen_img = model.gen.synthesis(new_latent, noise_mode='const')
    gen_img = gen_img.mul(0.5).add_(0.5)
    
    # Convert to PIL Image
    gen_img = gen_img.squeeze(0).detach().cpu()
    gen_img = gen_img.clamp(0, 1)
    
    # Convert to PIL Image
    img_array = gen_img.permute(1, 2, 0).numpy()
    img_array = (img_array * 255).astype(np.uint8)
    pil_image = Image.fromarray(img_array)
    
    if return_concepts:
        return pil_image, concept_values
    return pil_image


def generate_all_combinations(seed, num_concepts=8, dataset='cub',
                              expt_name='cbae_stygan2',
                              tensorboard_name='sup_pl_cls8_cbae.pt',
                              device='cuda:0'):
    """
    Generate all possible combinations for a given seed.
    
    Args:
        seed: Random seed for latent generation
        num_concepts: Number of concepts to vary (default: 8, gives 2^8=256 combinations)
        dataset: Dataset name
        expt_name: Experiment name
        tensorboard_name: Tensorboard checkpoint name
        device: Device to use
        
    Returns:
        dict: Dictionary mapping binary strings to PIL Images
    """
    # Initialize model
    model, (config, concept_names) = initialize_model(dataset, expt_name,
                                                       tensorboard_name, device)
    
    # Set random seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Generate latent vector from seed
    z = torch.randn((1, model.gen.z_dim), device=device)
    latent = model.gen.mapping(z, None, truncation_psi=1.0, truncation_cutoff=None)
    
    # Get original concepts
    concepts = model.cbae.enc(latent)
    
    # Generate all combinations
    all_combinations = list(itertools.product([0, 1], repeat=num_concepts))
    result_dict = {}
    
    print(f"Generating {len(all_combinations)} images for seed {seed}...")
    
    for combo in all_combinations:
        # Manipulate concepts
        new_concepts = manipulate_concepts(model, concepts, combo)
        
        # Decode and generate image
        new_latent = model.cbae.dec(new_concepts)
        gen_img = model.gen.synthesis(new_latent, noise_mode='const')
        gen_img = gen_img.mul(0.5).add_(0.5)
        
        # Convert to PIL Image
        gen_img = gen_img.squeeze(0).detach().cpu()
        gen_img = gen_img.clamp(0, 1)
        
        img_array = gen_img.permute(1, 2, 0).numpy()
        img_array = (img_array * 255).astype(np.uint8)
        pil_image = Image.fromarray(img_array)
        
        # Store with binary string as key
        binary_str = ''.join(map(str, combo))
        result_dict[binary_str] = pil_image
    
    print(f"Generated {len(result_dict)} images")
    return result_dict


def get_concept_names(dataset='cub', expt_name='cbae_stygan2'):
    """
    Get the concept names for a given dataset/experiment.
    
    Args:
        dataset: Dataset name
        expt_name: Experiment name
        
    Returns:
        list: List of concept names
    """
    config_file = project_root / f"config/{expt_name}/{dataset}.yaml"
    with open(config_file, 'r') as stream:
        config = yaml.safe_load(stream)
    
    # Return all concept names except 'unknown'
    return config['model']['concepts']['concept_names'][:-1]

