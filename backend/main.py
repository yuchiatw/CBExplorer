from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
from typing import List
import os
import io
import base64
from backend.CE_main import (
    get_concept_names,
    sample_cbm,
    sample_cbm_opt,
)

# Initialize FastAPI app
app = FastAPI(title="CBExplorer API", version="1.0.0")

# Configure CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],  # Vite default port
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Base directory for images
BASE_IMAGE_DIR = Path("./posthoc-generative-cbm/CE_data")


@app.get("/")
async def root():
    """Health check endpoint"""
    return {"status": "ok", "message": "CBExplorer API is running"}


@app.get("/concepts/{experiment}/{dataset}")
async def get_concepts(experiment: str, dataset: str):
    """
    Get the list of concept names for a given experiment and dataset.
    
    Args:
        experiment: Name of the experiment (e.g., 'cbae_stygan2')
        dataset: Name of the dataset (e.g., 'cub', 'celebahq')
        
    Returns:
        Dictionary with concept names
    """
    try:
        concept_names = get_concept_names(dataset=dataset, expt_name=experiment)
        
        return {
            "experiment": experiment,
            "dataset": dataset,
            "concepts": concept_names,
            "num_concepts": len(concept_names)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading concepts: {str(e)}")


def _process_generation_request(experiment: str, dataset: str, seed: int, bit: str = None):
    """
    Helper function to process generation requests.
    """
    try:
        # Validate bit string if provided
        if bit is not None:
            if not all(c in '01' for c in bit):
                raise HTTPException(status_code=400, detail="Bit string must contain only 0s and 1s")
            
        # Choose generator implementation based on experiment type
        experiment_lower = experiment.lower()
        device = 'cuda:0' if os.system('nvidia-smi > /dev/null 2>&1') == 0 else 'cpu'
        generator_kwargs = dict(
            seed=seed,
            combination=bit,
            dataset=dataset,
            expt_name=experiment,
            device=device,
        )
        
        if 'cc' in experiment_lower:
            generator_fn = sample_cbm_opt
        elif 'cbae' in experiment_lower:
            generator_fn = sample_cbm
        else:
            raise HTTPException(
                status_code=400,
                detail="Experiment must include 'cc' or 'cbae' to determine generator",
            )
        
        print(f"Generating image for: {experiment}/{dataset}/seed={seed}/bit={bit} using {generator_fn.__name__}")
        pil_image, concept_values, concept_probs = generator_fn(**generator_kwargs)
        
        # Convert PIL image to base64
        img_byte_arr = io.BytesIO()
        pil_image.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)
        img_base64 = base64.b64encode(img_byte_arr.read()).decode('utf-8')
        
        # Return JSON response with image and concept values
        return JSONResponse(content={
            "image": f"data:image/png;base64,{img_base64}",
            "concept_values": concept_values,
            "concept_probs": concept_probs,
            "requested_combination": bit,
            "seed": seed,
            "experiment": experiment,
            "dataset": dataset
        })
        
    except ImportError as e:
        raise HTTPException(status_code=500, detail=f"Model loading error: {str(e)}")
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Image generation error: {str(e)}")


@app.get("/generate/{experiment}/{dataset}/{seed}")
async def generate_sample(experiment: str, dataset: str, seed: int):
    """
    Dynamically generate an image using the CB model without concept intervention.
    Returns both the image (as base64) and the concept values (argmax for each concept).
    
    Args:
        experiment: Name of the experiment (e.g., 'cbae_stygan2', 'cc_stygan2')
        dataset: Name of the dataset (e.g., 'cub', 'celebahq')
        seed: Seed number for latent generation
        
    Returns:
        JSON response with image, concept values, and probabilities.
    """
    return _process_generation_request(experiment, dataset, seed)


@app.get("/manipulate/{experiment}/{dataset}/{seed}")
async def manipulate_sample(experiment: str, dataset: str, seed: int, bit: str):
    """
    Dynamically generate an image using the CB model with concept intervention.
    Returns both the image (as base64) and the concept values (argmax for each concept).
    
    Args:
        experiment: Name of the experiment (e.g., 'cbae_stygan2', 'cc_stygan2')
        dataset: Name of the dataset (e.g., 'cub', 'celebahq')
        seed: Seed number for latent generation
        bit: Binary string representing concept combination (e.g., "01101011")
        
    Returns:
        JSON response with image, concept values, and probabilities.
    """
    return _process_generation_request(experiment, dataset, seed, bit)
