"""
Main module for Concept Exploration (CE) image generation.
Provides API functions to dynamically generate images from the CB-AE model.
"""

import yaml
import torch
import numpy as np
import itertools
from pathlib import Path
from PIL import Image

config_root = Path(__file__).parent / "posthoc_generative_cbm" / "config"
checkpoints_root = Path(__file__).parent / "posthoc_generative_cbm" / "models" / "checkpoints"

from backend.posthoc_generative_cbm.models import cbae_stygan2
from backend.posthoc_generative_cbm.CE_utils import (
    get_concept_index,
    opt_int_multi,
)


# Global model cache to avoid reloading
_model_cache = {}
_config_cache = {}


def initialize_model(dataset='cub', expt_name='cbae_stygan2', device='cuda:0'):
    """
    Initialize and load the CB-AE/CC model. Uses caching to avoid reloading.
    
    Args:
        dataset: Dataset name (e.g., 'cub', 'celebahq')
        expt_name: Experiment name
        device: Device to load model on
        
    Returns:
        tuple: (model, (config, concept_names))
    """
    cache_key = f"{dataset}_{expt_name}"
    expt_name_lower = expt_name.lower()
    
    # Return cached model if available
    if cache_key in _model_cache:
        print(f"Using cached model for {cache_key}")
        return _model_cache[cache_key], _config_cache[cache_key]
    
    print(f"Loading model for {cache_key}...")
    
    # Load configuration relative to project root without mutating CWD
    config_file = config_root / expt_name / f"{dataset}.yaml"
    with open(config_file, 'r') as stream:
        config = yaml.safe_load(stream)
    print(f"Loaded configuration file {config_file}")
    
    # Validate configuration and extract concept names (use only binary concepts)
    concept_cfg = config['model']['concepts']
    concept_meta = list(zip(concept_cfg['concept_names'], concept_cfg['types']))
    concept_names = [name for name, concept_type in concept_meta if concept_type == 'bin']
    if not concept_names:
        raise ValueError("Configuration must define at least one binary concept.")
    print(f"Concept names: {concept_names}")
    
    # Set up model (CC uses encoder-only classifier, CB-AE includes decoder)
    if 'cbae' in expt_name_lower:
        model_cls = cbae_stygan2.cbAE_StyGAN2
    elif 'cc' in expt_name_lower:
        model_cls = cbae_stygan2.CC_StyGAN2
    else:
        raise ValueError(f"Unsupported experiment type for expt_name={expt_name}")
    print(f"Instantiating {model_cls.__name__} for {cache_key}")
    model = model_cls(config)
    
    # Load checkpoint without changing directories
    cbae_ckpt_path = checkpoints_root / f'{dataset}_{expt_name}.pt'
    model.cbae.load_state_dict(torch.load(str(cbae_ckpt_path), map_location='cpu'))
    
    # Move to device and set to evaluation mode
    model.to(device)
    model.eval()
    
    print(f"Model loaded successfully on {device}")
    
    # Cache the model and config
    _model_cache[cache_key] = model
    _config_cache[cache_key] = (config, concept_names)
    
    return model, (config, concept_names)


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


def _normalize_combination(combination):
    """
    Normalize concept combination input into a tuple of ints.
    """
    if isinstance(combination, str):
        if not set(combination).issubset({'0', '1'}):
            raise ValueError("Combination string must only contain 0 or 1.")
        return tuple(int(c) for c in combination)
    if isinstance(combination, (list, tuple)):
        return tuple(int(c) for c in combination)
    raise TypeError("Combination must be a string, list, or tuple of ints.")


def _sample_latent_for_seed(model, seed, device):
    """
    Sample a StyleGAN latent given a deterministic seed.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    z = torch.randn((1, model.gen.z_dim), device=device)
    return model.gen.mapping(z, None, truncation_psi=1.0, truncation_cutoff=None)


def _synthesize_latent_to_pil(model, latent):
    """
    Decode latent to image tensor and convert to PIL.Image.
    """
    gen_img = model.gen.synthesis(latent, noise_mode='const')
    gen_img = gen_img.mul(0.5).add_(0.5)
    gen_img = gen_img.squeeze(0).detach().cpu().clamp(0, 1)
    img_array = gen_img.permute(1, 2, 0).numpy()
    img_array = (img_array * 255).astype(np.uint8)
    return Image.fromarray(img_array)


def _extract_concept_values(model, concept_logits, num_concepts):
    """
    Convert concept logits into discrete concept predictions.
    """
    concept_values = []
    for concept_idx in range(num_concepts):
        start, end = get_concept_index(model, concept_idx)
        concept_slice = concept_logits[:, start:end]
        concept_values.append(int(torch.argmax(concept_slice, dim=1).item()))
    return concept_values


def generate_image_from_combination(seed, combination, dataset='cub', 
                                    expt_name='cbae_stygan2',
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
    model, _ = initialize_model(dataset, expt_name, device)
    
    # Normalize combination input
    combination = _normalize_combination(combination)
    num_concepts = len(combination)
    
    # Sample latent for seed
    latent = _sample_latent_for_seed(model, seed, device)
    
    # Get original concepts and adjust logits directly
    concepts = model.cbae.enc(latent)
    new_concepts = manipulate_concepts(model, concepts, combination)
    
    # Decode and synthesize image
    new_latent = model.cbae.dec(new_concepts)
    pil_image = _synthesize_latent_to_pil(model, new_latent)
    
    if return_concepts:
        concept_values = _extract_concept_values(model, new_concepts, num_concepts)
        return pil_image, concept_values
    return pil_image


def generate_image_from_combination_opt_int(seed, combination, dataset='cub',
                                            expt_name='cbae_stygan2',
                                            device='cuda:0',
                                            return_concepts=False,
                                            optint_iters=50,
                                            optint_eps=1e-1):
    """
    Generate an image for a specific seed and concept combination using
    optimization-based interventions (opt_int) on the latent space.
    
    Args:
        seed: Random seed for latent generation.
        combination: Binary combination string or iterable (e.g., "01101011").
        dataset: Dataset name.
        expt_name: Experiment name.
        device: Device to use.
        return_concepts: If True, returns (PIL Image, concept_values list).
        optint_iters: Number of optimization iterations per concept.
        optint_eps: L-infinity norm bound for the opt_int updates.
    
    Returns:
        PIL Image object, or (PIL Image, concept_values list) if return_concepts=True.
    """
    model, _ = initialize_model(dataset, expt_name, device)
    
    # Normalize combination input
    combination = _normalize_combination(combination)
    num_concepts = len(combination)
    
    # Sample latent for seed
    latent = _sample_latent_for_seed(model, seed, device)
    
    # Apply multi-concept optimization in a single pass
    new_latent = opt_int_multi(
        model=model,
        latent=latent.clone(),
        concept_values=[int(val) for val in combination],
        num_iters=optint_iters,
        eps=optint_eps,
        device=device,
    ).detach()
    
    pil_image = _synthesize_latent_to_pil(model, new_latent)
    
    if return_concepts:
        updated_concepts = model.cbae.enc(new_latent)
        concept_values = _extract_concept_values(model, updated_concepts, num_concepts)
        return pil_image, concept_values
    return pil_image


def get_concept_names(dataset='cub', expt_name='cbae_stygan2'):
    """
    Get the concept names for a given dataset/experiment.
    
    Args:
        dataset: Dataset name
        expt_name: Experiment name
        
    Returns:
        list: List of concept names
    """
    config_file = config_root / expt_name / f"{dataset}.yaml"
    with open(config_file, 'r') as stream:
        config = yaml.safe_load(stream)
    
    concept_cfg = config['model']['concepts']
    concept_meta = zip(concept_cfg['concept_names'], concept_cfg['types'])
    return [name for name, concept_type in concept_meta if concept_type == 'bin']

