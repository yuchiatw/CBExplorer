"""
Main module for Concept Exploration (CE) image generation.
Provides API functions to dynamically generate images from the CB-AE model.
"""

import yaml
import torch
import numpy as np
import itertools
from torch.autograd import Variable
from pathlib import Path
from PIL import Image

config_root = Path(__file__).parent / "posthoc_generative_cbm" / "config"
checkpoints_root = Path(__file__).parent / "posthoc_generative_cbm" / "models" / "checkpoints"

from backend.posthoc_generative_cbm.models import cbae_stygan2
from backend.posthoc_generative_cbm.CE_utils import (
    get_concept_index,
)
from backend.posthoc_generative_cbm.eval.eval_intervention_gan import opt_int


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
    Convert concept logits into discrete concept predictions and probability distributions.
    """
    concept_values = []
    concept_probs = []
    for concept_idx in range(num_concepts):
        start, end = get_concept_index(model, concept_idx)
        concept_slice = concept_logits[:, start:end]
        
        # Calculate probabilities
        probs = torch.softmax(concept_slice, dim=1)
        
        # Get probability of class 1 (assuming binary concepts)
        prob_1 = probs[:, 1].item()
        concept_probs.append(prob_1)
        
        # Determine discrete value (1 if > 0.5 else 0)
        concept_values.append(1 if prob_1 > 0.5 else 0)
        
    return concept_values, concept_probs




def sample_cbm(seed, combination=None, dataset='cub', 
                                    expt_name='cbae_stygan2',
                                    device='cuda:0'):
    """
    Generate an image for a specific seed and concept combination.
    
    Args:
        seed: Random seed for latent generation
        combination: Binary combination string (e.g., "01101011") or tuple (0,1,1,0,1,0,1,1). If None, no intervention.
        dataset: Dataset name
        expt_name: Experiment name
        device: Device to use
        
    Returns:
        tuple: (PIL Image, concept_values list, concept_probs list)
    """
    # Initialize model (uses cache if already loaded)
    model, (_, concept_names) = initialize_model(dataset, expt_name, device)
    
    num_concepts = len(concept_names)
    if combination is not None:
        # Normalize combination input
        combination = _normalize_combination(combination)
    
    # Sample latent for seed
    latent = _sample_latent_for_seed(model, seed, device)
    
    # Get original concepts and adjust logits directly
    concepts = model.cbae.enc(latent)
    
    if combination is not None:
        new_concepts = manipulate_concepts(model, concepts, combination)
    else:
        new_concepts = concepts
    
    # Decode and synthesize image
    new_latent = model.cbae.dec(new_concepts)
    pil_image = _synthesize_latent_to_pil(model, new_latent)
    
    concept_values, concept_probs = _extract_concept_values(model, new_concepts, num_concepts)
    return pil_image, concept_values, concept_probs


def sample_cbm_opt(seed, combination=None, dataset='cub',
                                            expt_name='cbae_stygan2',
                                            device='cuda:0',
                                            optint_iters=1000,
                                            optint_eps=1e-1):
    """
    Generate an image for a specific seed and concept combination using
    optimization-based interventions (opt_int) on the latent space.
    
    Args:
        seed: Random seed for latent generation.
        combination: Binary combination string or iterable (e.g., "01101011"). If None, no intervention.
        dataset: Dataset name.
        expt_name: Experiment name.
        device: Device to use.
        optint_iters: Number of optimization iterations per concept.
        optint_eps: L-infinity norm bound for the opt_int updates.
    
    Returns:
        tuple: (PIL Image, concept_values list, concept_probs list)
    """
    model, (_, concept_names) = initialize_model(dataset, expt_name, device)
    
    num_concepts = len(concept_names)
    if combination is not None:
        # Normalize combination input
        combination = _normalize_combination(combination)
    
    # Sample latent for seed
    latent = _sample_latent_for_seed(model, seed, device)
    
    # Apply optimization if combination is provided
    if combination is not None:
        # Get original concept values
        with torch.no_grad():
            original_concepts_logits = model.cbae.enc(latent)
            orig_values, _ = _extract_concept_values(model, original_concepts_logits, num_concepts)
        
        # Identify changing concepts (just for logging, we optimize all)
        diff_indices = [i for i, (a, b) in enumerate(zip(orig_values, combination)) if a != b]
        
        
        # Optimization setup
        noise = torch.zeros_like(latent, requires_grad=True, device=device)
        criterion = torch.nn.MSELoss()

        # PGD parameters
        alpha = 2.5 * optint_eps / optint_iters
        
        for i in range(optint_iters):
            if noise.grad is not None:
                noise.grad.zero_()
            
            adv_latent = latent + noise
            current_logits = model.cbae.enc(adv_latent)
            
            loss = 0.0
            # Optimize for ALL concepts in the combination
            for idx in range(num_concepts):
                start, end = get_concept_index(model, idx)
                
                # Construct target for MSE loss (one-hot-like)
                num_cls = end - start
                target_tensor = torch.zeros((latent.shape[0], num_cls), device=device)
                target_val_idx = combination[idx]
                target_tensor[:, target_val_idx] = 1.0
                
                loss += criterion(current_logits[:, start:end], target_tensor)
            
            loss.backward()
            
            with torch.no_grad():
                # Gradient descent on the loss (minimize MSE)
                noise_grad = noise.grad.sign()
                noise -= alpha * noise_grad
                noise = torch.clamp(noise, -optint_eps, optint_eps)
                noise.requires_grad = True
                
        new_latent = latent + noise.detach()
        
        with torch.no_grad():
            final_logits = model.cbae.enc(new_latent)
            final_loss = 0.0
            for idx in range(num_concepts):
                start, end = get_concept_index(model, idx)
                
                num_cls = end - start
                target_tensor = torch.zeros((latent.shape[0], num_cls), device=device)
                target_val_idx = combination[idx]
                target_tensor[:, target_val_idx] = 1.0
                
                final_loss += criterion(final_logits[:, start:end], target_tensor).item()
        print(f"Optimization finished. Final MSE Loss on all concepts: {final_loss:.6f}")
            
    else:
        new_latent = latent
    
    old_concepts = model.cbae.enc(latent)
    new_concepts = model.cbae.enc(new_latent)

    
    pil_image = _synthesize_latent_to_pil(model, new_latent)
    
    updated_concepts = model.cbae.enc(new_latent)
    concept_values, concept_probs = _extract_concept_values(model, updated_concepts, num_concepts)
    return pil_image, concept_values, concept_probs


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

