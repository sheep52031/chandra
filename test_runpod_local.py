#!/usr/bin/env python3
"""
Local test script for RunPod handler
This script allows you to test the handler locally before deploying to RunPod
"""

import base64
import sys
from pathlib import Path
from runpod_handler import handler


def test_with_local_image(image_path: str):
    """Test handler with a local image file."""
    print(f"Testing with local image: {image_path}")

    # Read and encode image
    with open(image_path, "rb") as f:
        image_data = f.read()
        image_base64 = base64.b64encode(image_data).decode("utf-8")

    # Create test event
    event = {
        "input": {
            "image": image_base64,
            "max_output_tokens": 12384,
            "include_images": True,
            "include_headers_footers": False,
            "prompt_type": "ocr_layout"
        }
    }

    # Run handler
    print("\nRunning handler...")
    result = handler(event)

    # Print results
    if "error" in result:
        print(f"\nError: {result['error']}")
        return False
    else:
        print("\nSuccess!")
        print(f"Number of results: {len(result['results'])}")
        for i, res in enumerate(result['results']):
            print(f"\n--- Result {i+1} ---")
            print(f"Token count: {res['token_count']}")
            print(f"Number of chunks: {len(res['chunks'])}")
            print(f"Number of images extracted: {len(res['images'])}")
            print(f"\nMarkdown preview (first 500 chars):")
            print(res['markdown'][:500])
            print("...")
        return True


def test_with_url(image_url: str):
    """Test handler with an image URL."""
    print(f"Testing with URL: {image_url}")

    # Create test event
    event = {
        "input": {
            "image": image_url,
            "max_output_tokens": 12384,
            "include_images": True,
            "include_headers_footers": False
        }
    }

    # Run handler
    print("\nRunning handler...")
    result = handler(event)

    # Print results
    if "error" in result:
        print(f"\nError: {result['error']}")
        return False
    else:
        print("\nSuccess!")
        print(f"Number of results: {len(result['results'])}")
        for i, res in enumerate(result['results']):
            print(f"\n--- Result {i+1} ---")
            print(f"Token count: {res['token_count']}")
            print(f"Number of chunks: {len(res['chunks'])}")
            print(f"\nMarkdown preview (first 500 chars):")
            print(res['markdown'][:500])
            print("...")
        return True


def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python test_runpod_local.py <image_path>")
        print("  python test_runpod_local.py --url <image_url>")
        print("\nExample:")
        print("  python test_runpod_local.py test_image.png")
        print("  python test_runpod_local.py --url https://example.com/document.png")
        sys.exit(1)

    if sys.argv[1] == "--url":
        if len(sys.argv) < 3:
            print("Error: URL not provided")
            sys.exit(1)
        success = test_with_url(sys.argv[2])
    else:
        image_path = sys.argv[1]
        if not Path(image_path).exists():
            print(f"Error: Image file not found: {image_path}")
            sys.exit(1)
        success = test_with_local_image(image_path)

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
