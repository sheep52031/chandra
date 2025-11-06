#!/usr/bin/env python3
"""
Example script for calling the deployed RunPod Serverless endpoint
"""

import runpod
import base64
import json
from pathlib import Path


def process_image_from_file(endpoint_id: str, api_key: str, image_path: str):
    """Process a local image file using the RunPod endpoint."""

    # Set API key
    runpod.api_key = api_key

    # Read and encode image
    with open(image_path, "rb") as f:
        image_base64 = base64.b64encode(f.read()).decode("utf-8")

    # Get endpoint
    endpoint = runpod.Endpoint(endpoint_id)

    # Run request
    print(f"Processing {image_path}...")
    result = endpoint.run({
        "input": {
            "image": image_base64,
            "max_output_tokens": 12384,
            "include_images": True,
            "include_headers_footers": False
        }
    })

    return result


def process_image_from_url(endpoint_id: str, api_key: str, image_url: str):
    """Process an image from URL using the RunPod endpoint."""

    # Set API key
    runpod.api_key = api_key

    # Get endpoint
    endpoint = runpod.Endpoint(endpoint_id)

    # Run request
    print(f"Processing {image_url}...")
    result = endpoint.run({
        "input": {
            "image": image_url,
            "max_output_tokens": 12384,
            "include_images": True,
            "include_headers_footers": False
        }
    })

    return result


def process_multiple_images(endpoint_id: str, api_key: str, image_paths: list):
    """Process multiple images using the RunPod endpoint."""

    # Set API key
    runpod.api_key = api_key

    # Read and encode all images
    images_base64 = []
    for image_path in image_paths:
        with open(image_path, "rb") as f:
            image_base64 = base64.b64encode(f.read()).decode("utf-8")
            images_base64.append(image_base64)

    # Get endpoint
    endpoint = runpod.Endpoint(endpoint_id)

    # Run request
    print(f"Processing {len(image_paths)} images...")
    result = endpoint.run({
        "input": {
            "images": images_base64,
            "max_output_tokens": 12384,
            "include_images": True,
            "include_headers_footers": False
        }
    })

    return result


def async_process_image(endpoint_id: str, api_key: str, image_path: str):
    """Process image asynchronously using the RunPod endpoint."""

    # Set API key
    runpod.api_key = api_key

    # Read and encode image
    with open(image_path, "rb") as f:
        image_base64 = base64.b64encode(f.read()).decode("utf-8")

    # Get endpoint
    endpoint = runpod.Endpoint(endpoint_id)

    # Run async request
    print(f"Starting async processing of {image_path}...")
    run_request = endpoint.run_async({
        "input": {
            "image": image_base64,
            "max_output_tokens": 12384,
            "include_images": True,
            "include_headers_footers": False
        }
    })

    # Get the job ID
    job_id = run_request.job_id
    print(f"Job ID: {job_id}")

    # Wait for completion and get result
    print("Waiting for result...")
    result = run_request.output()

    return result


def save_results(result: dict, output_dir: str = "output"):
    """Save the OCR results to files."""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    if "error" in result:
        print(f"Error: {result['error']}")
        return

    results = result.get("results", [])

    for i, res in enumerate(results):
        # Save markdown
        md_file = output_path / f"result_{i+1}.md"
        with open(md_file, "w", encoding="utf-8") as f:
            f.write(res["markdown"])
        print(f"Saved markdown to {md_file}")

        # Save HTML
        html_file = output_path / f"result_{i+1}.html"
        with open(html_file, "w", encoding="utf-8") as f:
            f.write(res["html"])
        print(f"Saved HTML to {html_file}")

        # Save extracted images
        for img_name, img_base64 in res["images"].items():
            img_data = base64.b64decode(img_base64)
            img_file = output_path / img_name
            with open(img_file, "wb") as f:
                f.write(img_data)
            print(f"Saved image to {img_file}")

        # Save metadata
        metadata = {
            "token_count": res["token_count"],
            "page_box": res["page_box"],
            "num_chunks": len(res["chunks"]),
            "num_images": len(res["images"])
        }
        metadata_file = output_path / f"result_{i+1}_metadata.json"
        with open(metadata_file, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
        print(f"Saved metadata to {metadata_file}")


if __name__ == "__main__":
    # Configuration
    RUNPOD_API_KEY = "YOUR_RUNPOD_API_KEY"  # Replace with your API key
    ENDPOINT_ID = "YOUR_ENDPOINT_ID"  # Replace with your endpoint ID

    # Example 1: Process single image from file
    print("=== Example 1: Single image from file ===")
    result = process_image_from_file(
        endpoint_id=ENDPOINT_ID,
        api_key=RUNPOD_API_KEY,
        image_path="test_image.png"
    )
    save_results(result, output_dir="output/example1")

    # Example 2: Process image from URL
    print("\n=== Example 2: Image from URL ===")
    result = process_image_from_url(
        endpoint_id=ENDPOINT_ID,
        api_key=RUNPOD_API_KEY,
        image_url="https://example.com/document.png"
    )
    save_results(result, output_dir="output/example2")

    # Example 3: Process multiple images
    print("\n=== Example 3: Multiple images ===")
    result = process_multiple_images(
        endpoint_id=ENDPOINT_ID,
        api_key=RUNPOD_API_KEY,
        image_paths=["page1.png", "page2.png", "page3.png"]
    )
    save_results(result, output_dir="output/example3")

    # Example 4: Async processing
    print("\n=== Example 4: Async processing ===")
    result = async_process_image(
        endpoint_id=ENDPOINT_ID,
        api_key=RUNPOD_API_KEY,
        image_path="test_image.png"
    )
    save_results(result, output_dir="output/example4")

    print("\nAll examples completed!")
