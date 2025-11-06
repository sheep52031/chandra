"""
RunPod Serverless Handler for Chandra OCR

This handler supports both image URLs and base64-encoded images.
"""

import runpod
import base64
import io
import os
from PIL import Image
from typing import Dict, Any, List
import requests
from chandra.model import InferenceManager
from chandra.model.schema import BatchInputItem


# Initialize the model globally to reuse across requests
print("Loading Chandra model...")
model = InferenceManager(method="hf")
print("Model loaded successfully!")


def download_image(url: str) -> Image.Image:
    """Download image from URL."""
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    return Image.open(io.BytesIO(response.content)).convert("RGB")


def decode_base64_image(base64_string: str) -> Image.Image:
    """Decode base64 string to PIL Image."""
    # Remove data URL prefix if present
    if "," in base64_string:
        base64_string = base64_string.split(",", 1)[1]

    image_data = base64.b64decode(base64_string)
    return Image.open(io.BytesIO(image_data)).convert("RGB")


def process_image_input(image_input: Any) -> Image.Image:
    """Process various image input formats."""
    if isinstance(image_input, str):
        if image_input.startswith("http://") or image_input.startswith("https://"):
            return download_image(image_input)
        else:
            # Assume base64
            return decode_base64_image(image_input)
    else:
        raise ValueError("Invalid image input format. Expected URL string or base64 string.")


def handler(event: Dict[str, Any]) -> Dict[str, Any]:
    """
    RunPod handler function for Chandra OCR.

    Expected input format:
    {
        "input": {
            "image": "URL or base64 string",  # Required: single image
            # OR
            "images": ["URL1", "URL2", ...],  # Required: multiple images

            # Optional parameters:
            "max_output_tokens": 12384,
            "include_images": true,
            "include_headers_footers": false,
            "prompt_type": "ocr_layout",  # or custom prompt
            "custom_prompt": "Custom prompt text"  # if not using prompt_type
        }
    }

    Returns:
    {
        "results": [
            {
                "markdown": "...",
                "html": "...",
                "chunks": [...],
                "raw": "...",
                "page_box": [x, y, width, height],
                "token_count": 1234,
                "images": {...},
                "error": false
            },
            ...
        ]
    }
    """
    try:
        job_input = event.get("input", {})

        # Get images input
        images_data = []
        if "image" in job_input:
            images_data = [job_input["image"]]
        elif "images" in job_input:
            images_data = job_input["images"]
        else:
            return {
                "error": "No 'image' or 'images' field provided in input"
            }

        # Parse optional parameters
        max_output_tokens = job_input.get("max_output_tokens", None)
        include_images = job_input.get("include_images", True)
        include_headers_footers = job_input.get("include_headers_footers", False)
        prompt_type = job_input.get("prompt_type", "ocr_layout")
        custom_prompt = job_input.get("custom_prompt", None)

        # Process images
        print(f"Processing {len(images_data)} image(s)...")
        pil_images = []
        for i, img_data in enumerate(images_data):
            try:
                pil_image = process_image_input(img_data)
                pil_images.append(pil_image)
                print(f"Loaded image {i+1}/{len(images_data)}: {pil_image.size}")
            except Exception as e:
                return {
                    "error": f"Failed to process image {i+1}: {str(e)}"
                }

        # Create batch input items
        batch = []
        for pil_image in pil_images:
            batch_item = BatchInputItem(
                image=pil_image,
                prompt_type=prompt_type,
                prompt=custom_prompt
            )
            batch.append(batch_item)

        # Run inference
        print("Running inference...")
        generate_kwargs = {
            "include_images": include_images,
            "include_headers_footers": include_headers_footers,
        }

        if max_output_tokens is not None:
            generate_kwargs["max_output_tokens"] = max_output_tokens

        results = model.generate(batch, **generate_kwargs)

        # Convert results to serializable format
        output_results = []
        for result in results:
            # Convert images to base64 for transmission
            images_base64 = {}
            for img_name, pil_img in result.images.items():
                buffer = io.BytesIO()
                pil_img.save(buffer, format="PNG")
                img_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
                images_base64[img_name] = img_base64

            output_results.append({
                "markdown": result.markdown,
                "html": result.html,
                "chunks": result.chunks,
                "raw": result.raw,
                "page_box": result.page_box,
                "token_count": result.token_count,
                "images": images_base64,
                "error": result.error,
            })

        print(f"Successfully processed {len(output_results)} image(s)")

        return {
            "results": output_results
        }

    except Exception as e:
        print(f"Error in handler: {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            "error": str(e)
        }


if __name__ == "__main__":
    # Start the RunPod serverless handler
    runpod.serverless.start({"handler": handler})
