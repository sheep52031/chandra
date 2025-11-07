"""
RunPod Serverless Handler for Chandra OCR

Supports:
- Images: PNG, JPG, JPEG, WebP, TIFF, BMP (URL or base64)
- PDFs: Single or multi-page (URL or base64)
"""

import runpod
import base64
import io
import os
import tempfile
import filetype
from PIL import Image
from typing import Dict, Any, List
import requests
from chandra.model import InferenceManager
from chandra.model.schema import BatchInputItem
from chandra.input import load_pdf_images


# Initialize the model globally to reuse across requests
print("Loading Chandra model...")
model = InferenceManager(method="hf")
print("Model loaded successfully!")


def download_file(url: str) -> bytes:
    """Download file from URL and return bytes."""
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    return response.content


def decode_base64_data(base64_string: str) -> bytes:
    """Decode base64 string to bytes."""
    # Remove data URL prefix if present (e.g., "data:application/pdf;base64,...")
    if "," in base64_string and base64_string.startswith("data:"):
        base64_string = base64_string.split(",", 1)[1]

    return base64.b64decode(base64_string)


def process_pdf_bytes(pdf_bytes: bytes, page_range: List[int] = None) -> List[Image.Image]:
    """Process PDF bytes and return list of PIL Images."""
    # Save to temporary file
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp_file:
        tmp_file.write(pdf_bytes)
        tmp_path = tmp_file.name

    try:
        # Use Chandra's PDF processing
        # IMPORTANT: Pass None (not []) when page_range is not specified
        # Empty list [] means "process zero pages", None means "process all pages"
        images = load_pdf_images(tmp_path, page_range if page_range else None)
        return images
    finally:
        # Clean up temporary file
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def process_image_bytes(image_bytes: bytes) -> Image.Image:
    """Process image bytes and return PIL Image."""
    return Image.open(io.BytesIO(image_bytes)).convert("RGB")


def process_file_input(file_input: str, page_range: List[int] = None) -> List[Image.Image]:
    """
    Process file input (URL or base64) and return list of PIL Images.

    Supports:
    - Image formats: PNG, JPG, JPEG, WebP, TIFF, BMP, GIF
    - PDF: Returns multiple images (one per page)
    """
    # Download or decode file
    if file_input.startswith("http://") or file_input.startswith("https://"):
        print(f"Downloading file from URL...")
        file_bytes = download_file(file_input)
    else:
        print(f"Decoding base64 data...")
        file_bytes = decode_base64_data(file_input)

    # Detect file type
    file_type = filetype.guess(file_bytes)

    if file_type is None:
        raise ValueError("Unable to determine file type. Please provide a valid image or PDF.")

    print(f"Detected file type: {file_type.extension}")

    # Process based on file type
    if file_type.extension == "pdf":
        print(f"Processing PDF (page_range: {page_range or 'all'})...")
        return process_pdf_bytes(file_bytes, page_range)
    elif file_type.extension in ["png", "jpg", "jpeg", "webp", "tiff", "bmp", "gif"]:
        print(f"Processing image...")
        return [process_image_bytes(file_bytes)]
    else:
        raise ValueError(f"Unsupported file type: {file_type.extension}")


def handler(event: Dict[str, Any]) -> Dict[str, Any]:
    """
    RunPod handler function for Chandra OCR.

    Expected input format:
    {
        "input": {
            "file": "URL or base64 string",  # Single file (image or PDF)
            # OR
            "files": ["URL1", "URL2", ...],  # Multiple files

            # Legacy support (deprecated)
            "image": "URL or base64 string",
            "images": ["URL1", "URL2", ...],

            # Optional parameters:
            "page_range": [0, 1, 2],  # For PDFs: which pages to process (0-indexed)
            "max_output_tokens": 12384,
            "include_images": true,
            "include_headers_footers": false,
            "prompt_type": "ocr_layout",
            "custom_prompt": "Custom prompt text"
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
                "images": {...},  # base64 encoded extracted images
                "error": false
            },
            ...
        ]
    }
    """
    try:
        job_input = event.get("input", {})

        # Get file input (support both new and legacy formats)
        files_data = []
        if "file" in job_input:
            files_data = [job_input["file"]]
        elif "files" in job_input:
            files_data = job_input["files"]
        elif "image" in job_input:  # Legacy support
            files_data = [job_input["image"]]
        elif "images" in job_input:  # Legacy support
            files_data = job_input["images"]
        else:
            return {
                "error": "No 'file', 'files', 'image', or 'images' field provided in input"
            }

        # Parse optional parameters
        page_range = job_input.get("page_range", None)
        max_output_tokens = job_input.get("max_output_tokens", None)
        include_images = job_input.get("include_images", True)
        include_headers_footers = job_input.get("include_headers_footers", False)
        prompt_type = job_input.get("prompt_type", "ocr_layout")
        custom_prompt = job_input.get("custom_prompt", None)

        # Process all files and collect images
        print(f"Processing {len(files_data)} file(s)...")
        all_images = []

        for i, file_data in enumerate(files_data):
            try:
                images = process_file_input(file_data, page_range)
                all_images.extend(images)
                print(f"File {i+1}/{len(files_data)}: Loaded {len(images)} image(s)")
            except Exception as e:
                return {
                    "error": f"Failed to process file {i+1}: {str(e)}"
                }

        print(f"Total images to process: {len(all_images)}")

        # Run inference on each page individually with progress logging
        print(f"Running OCR inference on {len(all_images)} page(s)...")
        generate_kwargs = {
            "include_images": include_images,
            "include_headers_footers": include_headers_footers,
        }

        if max_output_tokens is not None:
            generate_kwargs["max_output_tokens"] = max_output_tokens

        results = []
        for idx, pil_image in enumerate(all_images):
            print(f"Processing page {idx + 1}/{len(all_images)}...")

            # Create single-item batch
            batch_item = BatchInputItem(
                image=pil_image,
                prompt_type=prompt_type,
                prompt=custom_prompt
            )

            # Process single page
            page_results = model.generate([batch_item], **generate_kwargs)
            results.extend(page_results)

            print(f"✓ Page {idx + 1}/{len(all_images)} completed ({page_results[0].token_count} tokens)")

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

        print(f"✅ Successfully processed {len(output_results)} page(s)")

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
    print("🚀 Starting RunPod serverless handler...")
    runpod.serverless.start({"handler": handler})
