from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
from typing import List
import os
import io
import sys
import base64

# Add the posthoc-generative-cbm directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'posthoc-generative-cbm'))

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
        from CE.CE_main import get_concept_names
        
        concept_names = get_concept_names(dataset=dataset, expt_name=experiment)
        
        return {
            "experiment": experiment,
            "dataset": dataset,
            "concepts": concept_names,
            "num_concepts": len(concept_names)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading concepts: {str(e)}")

# # Original static-only endpoint (kept for reference)
# @app.get("/image/{experiment}/{dataset}/{seed}")
# async def get_image(experiment: str, dataset: str, seed: int, bit: str):
#     """
#     Get an image from the specified experiment, dataset, and seed
#     
#     Args:
#         experiment: Name of the experiment
#         dataset: Name of the dataset
#         seed: Seed number
#     """
#     image_path = BASE_IMAGE_DIR / f"{experiment}-{dataset}" / str(seed) / f"{bit}.png"
#     print(image_path)
#     if not image_path.exists() or not image_path.is_file():
#         raise HTTPException(status_code=404, detail="Image not found")
#     return FileResponse(image_path)


@app.get("/generate/{experiment}/{dataset}/{seed}")
async def generate_image(experiment: str, dataset: str, seed: int, bit: str):
    """
    Dynamically generate an image using the CB-AE model.
    Returns both the image (as base64) and the concept values (argmax for each concept).
    
    Args:
        experiment: Name of the experiment (e.g., 'cbae_stygan2')
        dataset: Name of the dataset (e.g., 'cub', 'celebahq')
        seed: Seed number for latent generation
        bit: Binary string representing concept combination (e.g., "01101011")
        
    Returns:
        JSON response with:
        - image: base64-encoded PNG image
        - concept_values: list of argmax values for each concept
        - requested_combination: the input bit string
        - seed: the seed used
    """
    try:
        # Import the generation module
        from CE.CE_main import generate_image_from_combination
        
        # Validate bit string
        if not all(c in '01' for c in bit):
            raise HTTPException(status_code=400, detail="Bit string must contain only 0s and 1s")
        
        # Map experiment name to tensorboard name (you may need to adjust this mapping)
        tensorboard_mapping = {
            'cbae_stygan2': 'sup_pl_cls8_cbae.pt',
            'cbae_stygan2_thr90': 'sup_pl_cls8_cbae.pt',
        }
        
        tensorboard_name = tensorboard_mapping.get(experiment, 'sup_pl_cls8_cbae.pt')
        
        # Generate the image with concept values
        print(f"Generating image for: {experiment}/{dataset}/seed={seed}/bit={bit}")
        pil_image, concept_values = generate_image_from_combination(
            seed=seed,
            combination=bit,
            dataset=dataset,
            expt_name=experiment,
            tensorboard_name=tensorboard_name,
            device='cuda:0' if os.system('nvidia-smi > /dev/null 2>&1') == 0 else 'cpu',
            return_concepts=True
        )
        
        # Convert PIL image to base64
        img_byte_arr = io.BytesIO()
        pil_image.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)
        img_base64 = base64.b64encode(img_byte_arr.read()).decode('utf-8')
        
        # Return JSON response with image and concept values
        return JSONResponse(content={
            "image": f"data:image/png;base64,{img_base64}",
            "concept_values": concept_values,
            "requested_combination": bit,
            "seed": seed,
            "experiment": experiment,
            "dataset": dataset
        })
        
    except ImportError as e:
        raise HTTPException(status_code=500, detail=f"Model loading error: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Image generation error: {str(e)}")







# @app.get("/api/sets")
# async def get_available_sets() -> List[str]:
#     """Get list of available image sets"""
#     try:
#         if not BASE_IMAGE_DIR.exists():
#             return []
        
#         sets = [d.name for d in BASE_IMAGE_DIR.iterdir() if d.is_dir()]
#         return sets
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Error reading sets: {str(e)}")


# @app.get("/api/images/{set_name}/{filename}")
# async def get_image(set_name: str, filename: str):
#     """
#     Get a specific image from a set
    
#     Args:
#         set_name: Name of the image set directory
#         filename: Image filename (e.g., '00000000.png')
    
#     Returns:
#         Image file
#     """
#     try:
#         # Construct the full path
#         image_path = BASE_IMAGE_DIR / set_name / filename
        
#         # Security check: ensure the path is within BASE_IMAGE_DIR
#         if not image_path.resolve().is_relative_to(BASE_IMAGE_DIR.resolve()):
#             raise HTTPException(status_code=403, detail="Access denied")
        
#         # Check if file exists
#         if not image_path.exists() or not image_path.is_file():
#             raise HTTPException(status_code=404, detail="Image not found")
        
#         # Return the image file
#         return FileResponse(image_path)
    
#     except HTTPException:
#         raise
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Error serving image: {str(e)}")


# @app.get("/api/images/{set_name}")
# async def get_images_in_set(set_name: str) -> List[str]:
#     """
#     Get list of all images in a specific set
    
#     Args:
#         set_name: Name of the image set directory
    
#     Returns:
#         List of image filenames
#     """
#     try:
#         set_path = BASE_IMAGE_DIR / set_name
        
#         # Check if set exists
#         if not set_path.exists() or not set_path.is_dir():
#             raise HTTPException(status_code=404, detail="Set not found")
        
#         # Get all image files (png, jpg, jpeg)
#         images = [
#             f.name for f in set_path.iterdir() 
#             if f.is_file() and f.suffix.lower() in ['.png', '.jpg', '.jpeg']
#         ]
        
#         return sorted(images)
    
#     except HTTPException:
#         raise
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Error reading images: {str(e)}")


# # Run with: uvicorn main:app --reload --host 0.0.0.0 --port 8000