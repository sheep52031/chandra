"""
RunPod Serverless Handler for Chandra OCR (Simplified Image-Only Version)

IMPORTANT ARCHITECTURE NOTE:
This handler ONLY accepts base64-encoded images. For PDF files, use client-side
conversion with Chandra's native load_pdf_images() function BEFORE sending to RunPod.

Supported Input Formats:
- Single image: {"image": "base64_string"}
- Multiple images: {"images": ["base64_1", "base64_2", ...]}
- Image URLs: {"image": "https://..."} or {"images": ["https://...", ...]}

For PDFs:
- Convert client-side using: chandra.input.load_pdf_images(pdf_path)
- Then send resulting PIL images as base64 to this handler
- See example: client_pdf_ocr.py

Output Format:
{
    "results": [
        {
            "page_number": 1,
            "markdown": "...",
            "html": "...",
            "chunks": [...],
            "raw": "...",
            "page_box": [x, y, width, height],
            "token_count": 1234,
            "images": {...},  # base64 encoded extracted images
            "error": false
        },
        ...
    ],
    "total_pages": N,
    "total_tokens": NNNN
}
"""

import runpod
import base64
import io
from PIL import Image
from typing import Dict, Any, List
import requests
from chandra.model import InferenceManager
from chandra.model.schema import BatchInputItem


# Initialize the model globally to reuse across requests
print("Loading Chandra model...")
model = InferenceManager(method="hf")
print("Model loaded successfully!")


def download_image(url: str) -> bytes:
    """Download image from URL and return bytes."""
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    return response.content


def decode_base64_image(base64_string: str) -> bytes:
    """Decode base64 string to bytes."""
    # Remove data URL prefix if present (e.g., "data:image/png;base64,...")
    if "," in base64_string and base64_string.startswith("data:"):
        base64_string = base64_string.split(",", 1)[1]

    return base64.b64decode(base64_string)


def process_image_input(image_input: str) -> Image.Image:
    """
    Process image input (URL or base64) and return PIL Image.

    Args:
        image_input: Either a URL (http://... or https://...) or base64-encoded image string

    Returns:
        PIL.Image: RGB image ready for OCR

    Raises:
        ValueError: If image cannot be decoded or loaded
    """
    # Download or decode image
    if image_input.startswith("http://") or image_input.startswith("https://"):
        print(f"Downloading image from URL...")
        image_bytes = download_image(image_input)
    else:
        print(f"Decoding base64 image data...")
        image_bytes = decode_base64_image(image_input)

    # Load as PIL Image
    try:
        pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        print(f"Image loaded: {pil_image.size[0]}x{pil_image.size[1]} pixels")
        return pil_image
    except Exception as e:
        raise ValueError(f"Failed to load image: {str(e)}")


def handler(event: Dict[str, Any]) -> Dict[str, Any]:
    """
    RunPod handler function for Chandra OCR (Image-Only).

    Expected input format:
    {
        "input": {
            # Single image (either format):
            "image": "URL or base64 string",

            # Multiple images (batch):
            "images": ["URL1 or base64", "URL2 or base64", ...],

            # Optional parameters:
            "max_output_tokens": 12384,
            "include_images": true,
            "include_headers_footers": false,
            "prompt_type": "ocr_layout",  # or "ocr_only"
            "custom_prompt": "Custom prompt text"
        }
    }

    Returns:
    {
        "results": [
            {
                "page_number": 1,
                "markdown": "...",
                "html": "...",
                "chunks": [...],
                "raw": "...",
                "page_box": [x, y, width, height],
                "token_count": 1234,
                "images": {...},  # base64 encoded extracted images
                "error": false
            },
            ...
        ],
        "total_pages": N,
        "total_tokens": NNNN
    }
    """
    try:
        job_input = event.get("input", {})

        # Get image input(s)
        images_data = []
        if "image" in job_input:
            images_data = [job_input["image"]]
        elif "images" in job_input:
            images_data = job_input["images"]
        else:
            return {
                "error": "No 'image' or 'images' field provided in input. "
                        "This handler only accepts images. For PDFs, convert to images "
                        "client-side using chandra.input.load_pdf_images() first."
            }

        # Parse optional parameters
        max_output_tokens = job_input.get("max_output_tokens", None)
        include_images = job_input.get("include_images", True)
        include_headers_footers = job_input.get("include_headers_footers", False)
        prompt_type = job_input.get("prompt_type", "ocr_layout")
        custom_prompt = job_input.get("custom_prompt", None)

        # Process all images
        print(f"Processing {len(images_data)} image(s)...")
        all_images = []

        for i, image_data in enumerate(images_data):
            try:
                pil_image = process_image_input(image_data)
                all_images.append(pil_image)
                print(f"Image {i+1}/{len(images_data)}: Loaded successfully")
            except Exception as e:
                return {
                    "error": f"Failed to process image {i+1}: {str(e)}"
                }

        print(f"Total images to process: {len(all_images)}")

        # Run inference on each image sequentially with progress logging
        print(f"Running OCR inference on {len(all_images)} image(s)...")
        generate_kwargs = {
            "include_images": include_images,
            "include_headers_footers": include_headers_footers,
        }

        if max_output_tokens is not None:
            generate_kwargs["max_output_tokens"] = max_output_tokens

        results = []
        for idx, pil_image in enumerate(all_images):
            print(f"Processing image {idx + 1}/{len(all_images)}...")

            # Create single-item batch
            batch_item = BatchInputItem(
                image=pil_image,
                prompt_type=prompt_type,
                prompt=custom_prompt
            )

            # Process single image
            page_results = model.generate([batch_item], **generate_kwargs)
            results.extend(page_results)

            print(f"✓ Image {idx + 1}/{len(all_images)} completed ({page_results[0].token_count} tokens)")

        # Convert results to serializable format
        output_results = []
        for idx, result in enumerate(results):
            # Convert images to base64 for transmission
            images_base64 = {}
            for img_name, pil_img in result.images.items():
                buffer = io.BytesIO()
                pil_img.save(buffer, format="PNG")
                img_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
                images_base64[img_name] = img_base64

            output_results.append({
                "page_number": idx + 1,
                "markdown": result.markdown,
                "html": result.html,
                "chunks": result.chunks,
                "raw": result.raw,
                "page_box": result.page_box,
                "token_count": result.token_count,
                "images": images_base64,
                "error": result.error,
            })

        print(f"✅ Successfully processed {len(output_results)} image(s)")

        return {
            "results": output_results,
            "total_pages": len(output_results),
            "total_tokens": sum(r["token_count"] for r in output_results)
        }

    except Exception as e:
        print(f"❌ Error in handler: {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            "error": str(e),
            "traceback": traceback.format_exc()
        }


if __name__ == "__main__":
    # Start the RunPod serverless handler
    print("🚀 Starting RunPod serverless handler (Image-Only Version)...")
    print("📌 For PDFs: Convert to images client-side using chandra.input.load_pdf_images()")
    runpod.serverless.start({"handler": handler})
