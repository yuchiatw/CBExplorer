"""
Utility functions for storing Concept Exploration (CE) data.
"""

import shutil
from pathlib import Path
from tqdm import tqdm
from torchvision.utils import save_image
import yaml
import torch
import numpy as np
from torch.autograd import Variable
from models import cbae_stygan2


def store_ce_images(
    all_combinations,
    all_images_tensor,
    expt_name,
    dataset,
    config_file,
    seed_num
):
    """
    Store generated concept exploration images with their binary labels.
    
    Args:
        all_combinations: List of tuples, each representing a binary combination of concepts
        all_images_tensor: Tensor containing all generated images
        expt_name: Experiment name (e.g., 'cbae_stygan2_thr90')
        dataset: Dataset name (e.g., 'celebahq')
        config_file: Path to the configuration file to copy
        seed_num: Seed number used for generation
        
    Returns:
        output_dir: Path object pointing to the created output directory
    """
    # Create output directory with seed number
    output_dir = Path(f"CE_data/{expt_name}-{dataset}") / str(seed_num)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Saving images to {output_dir}")
    
    # Save each image with its binary label as the filename
    for i, (combo, img) in enumerate(tqdm(zip(all_combinations, all_images_tensor), desc="Saving images")):
        # Convert combination tuple to binary string (e.g., (0,1,1,0,1,0,1,1) -> "01101011")
        binary_label = ''.join(map(str, combo))
        
        # Save the image
        img_path = output_dir / f"{binary_label}.png"
        save_image(img, img_path)
    
    print(f"Saved {len(all_images_tensor)} images")
    
    # Copy the config file to the data directory
    config_dest = output_dir / Path(config_file).name
    shutil.copy2(config_file, config_dest)
    print(f"Copied config file to {config_dest}")
    
    return output_dir


def load_cb_model(
    dataset,
    expt_name,
    tensorboard_name,
    device='cuda:0'
):
    """
    Load a CB-AE model with the specified configuration.
    
    Args:
        dataset: Dataset name (e.g., 'celebahq')
        expt_name: Experiment name (e.g., 'cbae_stygan2_thr90')
        tensorboard_name: Tensorboard name (e.g., 'sup_pl_unk40_cls8')
        device: Device to load the model on (default: 'cuda:0')
        
    Returns:
        tuple: (model, config, num_cls, conc_clsf_classes, config_file)
            - model: Loaded CB-AE StyleGAN2 model
            - config: Configuration dictionary
            - num_cls: Number of concept classes (excluding 'unknown')
            - conc_clsf_classes: List of concept names (excluding 'unknown')
            - config_file: Path to the configuration file
    """
    # Load configuration
    config_file = f"./config/{expt_name}/{dataset}.yaml"
    
    with open(config_file, 'r') as stream:
        config = yaml.safe_load(stream)
    print(f"Loaded configuration file {config_file}")
    
    # Validate configuration
    if config['model']['concepts']['concept_names'][-1] != 'unknown':
        raise ValueError("Last concept name must be 'unknown'")
    if not all([type == 'bin' for type in config['model']['concepts']['types'][:-1]]):
        raise ValueError("All but the last concept must be binary")
    
    # Extract concept information
    num_cls = len(config['model']['concepts']['concept_bins']) - 1
    conc_clsf_classes = config['model']['concepts']['concept_names'][:-1]
    
    print(f"num_cls: {num_cls}")
    print(f"conc_clsf_classes: {conc_clsf_classes}")
    
    # Set up model
    config['model']['pretrained'] = 'models/checkpoints/stylegan2-celebahq-256x256.pkl'
    model = cbae_stygan2.cbAE_StyGAN2(config)
    
    # Load checkpoint
    cbae_ckpt_path = f'models/checkpoints/{dataset}_{expt_name}_{tensorboard_name}'
    model.cbae.load_state_dict(torch.load(cbae_ckpt_path, map_location='cpu'))
    
    # Move to device and set to evaluation mode
    model.to(device)
    model.eval()
    
    return model, config, num_cls, conc_clsf_classes, config_file


def get_concept_index(model, c):
    """
    Get the start and end indices for a specific concept in the concept vector.
    
    Args:
        model: The CB-AE model
        c: Concept index
        
    Returns:
        tuple: (start_index, end_index)
    """
    if c == 0:
        start = 0
    else:
        start = sum(model.concept_bins[:c])
    end = sum(model.concept_bins[:c+1])
    
    return start, end


def opt_int_multi(model, latent, concept_values, num_iters=50, eps=1e-1, device='cuda:0'):
    """
    Optimization-based intervention for multiple concepts simultaneously.
    
    This function modifies a latent vector to achieve desired values for multiple concepts
    at the same time, using an iterative optimization approach.
    
    Args:
        model: CB-AE model with encoder (cbae.enc)
        latent: Input latent vector to be modified
        concept_values: List or array of desired concept values (e.g., [0, 1, 1, 0, 1, 0, 1, 1])
                       Length should match the number of concepts
        num_iters: Number of optimization iterations (default: 50)
        eps: L-infinity norm bound for the perturbation (default: 0.1)
        device: Device to run computations on (default: 'cuda:0')
        
    Returns:
        torch.Tensor: Modified latent vector with desired concept values
        
    Example:
        >>> # Optimize for all 8 concepts at once
        >>> new_latent = opt_int_multi(model, latent, 
        ...                            concept_values=[1, 0, 1, 1, 0, 1, 0, 1],
        ...                            num_iters=50, eps=0.1)
    """
    latent = Variable(latent, requires_grad=True)
    noise = torch.FloatTensor(np.random.uniform(-eps, eps, [*latent.shape])).to(device)
    ce_loss = torch.nn.CrossEntropyLoss()
    
    # Convert concept_values to tensor if it's a list/tuple
    if not isinstance(concept_values, torch.Tensor):
        concept_values = list(concept_values)
    
    # Build target labels for all concepts
    num_concepts = len(concept_values)
    total_loss_weight = 1.0 / num_concepts  # Normalize loss across concepts
    
    # Pre-build all labels for efficiency
    labels = []
    concept_ranges = []
    for concept_idx in range(num_concepts):
        start, end = get_concept_index(model, concept_idx)
        num_cls = end - start
        
        # Create one-hot label for this concept
        label = torch.zeros((latent.shape[0], num_cls)).to(device)
        label[:, concept_values[concept_idx]] = 1.0
        label = label.float()
        
        labels.append(label)
        concept_ranges.append((start, end))
    
    # Optimization loop
    for iter_idx in range(num_iters):
        adv_latent = latent + noise
        adv_latent = Variable(adv_latent, requires_grad=True)
        
        # Clear gradients
        if adv_latent.grad is not None:
            adv_latent.grad.detach_()
            adv_latent.grad.zero_()
        
        # Forward pass through encoder
        concepts = model.cbae.enc(adv_latent)
        
        # Compute combined loss for all concepts
        total_loss = 0.0
        for concept_idx in range(num_concepts):
            start, end = concept_ranges[concept_idx]
            # Minimize cross-entropy loss for each concept
            total_loss += -ce_loss(concepts[:, start:end], labels[concept_idx])
        
        # Average the loss
        total_loss = total_loss * total_loss_weight
        total_loss.backward()
        
        # Update noise using gradient sign
        grad_sign = eps * torch.sign(adv_latent.grad.data)
        noise = grad_sign.to(device)
        noise = torch.clamp(noise, -eps, eps)
    
    return latent + noise

