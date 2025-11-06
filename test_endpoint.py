#!/usr/bin/env python3
"""
Simple test script for RunPod Serverless endpoint
使用此脚本快速测试您的 RunPod Endpoint 是否正常工作
"""

import os
import sys
import runpod
import base64
import json
from pathlib import Path


def test_endpoint_with_image(api_key: str, endpoint_id: str, image_path: str):
    """
    Test RunPod endpoint with an image file.

    Args:
        api_key: RunPod API key
        endpoint_id: RunPod endpoint ID
        image_path: Path to test image
    """

    print("=" * 60)
    print("Testing RunPod Serverless Endpoint")
    print("=" * 60)
    print(f"\nEndpoint ID: {endpoint_id}")
    print(f"Image: {image_path}\n")

    # Set API key
    runpod.api_key = api_key

    # Read and encode image
    print("📁 Reading image...")
    try:
        with open(image_path, "rb") as f:
            image_base64 = base64.b64encode(f.read()).decode("utf-8")
        print("✓ Image loaded successfully")
    except FileNotFoundError:
        print(f"❌ Error: Image file not found: {image_path}")
        sys.exit(1)

    # Create endpoint
    print(f"\n🔗 Connecting to endpoint: {endpoint_id}...")
    endpoint = runpod.Endpoint(endpoint_id)

    # Send request
    print("📤 Sending request to RunPod...")
    print("   (This may take 10-30 seconds for the first request)\n")

    try:
        result = endpoint.run_sync(
            {
                "input": {
                    "image": image_base64,
                    "max_output_tokens": 12384,
                    "include_images": True,
                    "include_headers_footers": False
                }
            },
            timeout=300  # 5 minutes timeout
        )

        print("=" * 60)
        print("✅ Request completed successfully!")
        print("=" * 60)

        # Check for errors
        if isinstance(result, dict) and "error" in result:
            print(f"\n❌ Error from endpoint:")
            print(json.dumps(result, indent=2, ensure_ascii=False))
            return False

        # Display results
        if isinstance(result, dict) and "results" in result:
            results = result["results"]
            print(f"\n📊 Processed {len(results)} image(s)")

            for i, res in enumerate(results, 1):
                print(f"\n--- Result {i} ---")
                print(f"Token count: {res.get('token_count', 'N/A')}")
                print(f"Extracted images: {len(res.get('images', {}))}")
                print(f"Chunks: {len(res.get('chunks', []))}")

                # Display first 200 characters of markdown
                markdown = res.get('markdown', '')
                if markdown:
                    print(f"\nMarkdown preview (first 200 chars):")
                    print("-" * 60)
                    print(markdown[:200])
                    if len(markdown) > 200:
                        print("...")
                    print("-" * 60)

                # Save full result to file
                output_dir = Path("test_output")
                output_dir.mkdir(exist_ok=True)

                md_file = output_dir / f"result_{i}.md"
                with open(md_file, "w", encoding="utf-8") as f:
                    f.write(markdown)
                print(f"\n💾 Full markdown saved to: {md_file}")

                html_file = output_dir / f"result_{i}.html"
                with open(html_file, "w", encoding="utf-8") as f:
                    f.write(res.get('html', ''))
                print(f"💾 HTML saved to: {html_file}")

            print("\n" + "=" * 60)
            print("🎉 Test completed successfully!")
            print("=" * 60)
            return True

        else:
            print(f"\n⚠️ Unexpected result format:")
            print(json.dumps(result, indent=2, ensure_ascii=False))
            return False

    except Exception as e:
        print("\n" + "=" * 60)
        print("❌ Request failed!")
        print("=" * 60)
        print(f"\nError: {e}")
        print("\nTroubleshooting:")
        print("1. Check if endpoint ID is correct")
        print("2. Verify API key is valid")
        print("3. Ensure endpoint is running (check RunPod console)")
        print("4. Check endpoint logs for details")
        return False


def test_endpoint_with_url(api_key: str, endpoint_id: str, image_url: str):
    """
    Test RunPod endpoint with an image URL.

    Args:
        api_key: RunPod API key
        endpoint_id: RunPod endpoint ID
        image_url: URL of test image
    """

    print("=" * 60)
    print("Testing RunPod Serverless Endpoint with URL")
    print("=" * 60)
    print(f"\nEndpoint ID: {endpoint_id}")
    print(f"Image URL: {image_url}\n")

    # Set API key
    runpod.api_key = api_key

    # Create endpoint
    print(f"🔗 Connecting to endpoint: {endpoint_id}...")
    endpoint = runpod.Endpoint(endpoint_id)

    # Send request
    print("📤 Sending request to RunPod...")

    try:
        result = endpoint.run_sync(
            {
                "input": {
                    "image": image_url,
                    "max_output_tokens": 12384,
                    "include_images": True
                }
            },
            timeout=300
        )

        print("✅ Request completed!")
        print("\nResult:")
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return True

    except Exception as e:
        print(f"\n❌ Request failed: {e}")
        return False


def main():
    """Main test function."""

    # Get configuration from environment or command line
    api_key = os.getenv("RUNPOD_API_KEY")
    endpoint_id = os.getenv("RUNPOD_ENDPOINT_ID")

    if not api_key:
        print("❌ Error: RUNPOD_API_KEY environment variable not set")
        print("\nPlease set it:")
        print("  export RUNPOD_API_KEY='your_api_key'")
        print("  或者在 .env.runpod 文件中设置")
        sys.exit(1)

    if not endpoint_id:
        print("❌ Error: RUNPOD_ENDPOINT_ID environment variable not set")
        print("\nPlease set it:")
        print("  export RUNPOD_ENDPOINT_ID='your_endpoint_id'")
        print("  或者在 .env.runpod 文件中设置")
        sys.exit(1)

    # Remove quotes if present
    api_key = api_key.strip('"').strip("'")
    endpoint_id = endpoint_id.strip('"').strip("'")

    # Check for image path argument
    if len(sys.argv) > 1:
        image_input = sys.argv[1]

        # Check if it's a URL or file path
        if image_input.startswith("http://") or image_input.startswith("https://"):
            success = test_endpoint_with_url(api_key, endpoint_id, image_input)
        else:
            success = test_endpoint_with_image(api_key, endpoint_id, image_input)
    else:
        print("❌ Error: No image provided")
        print("\nUsage:")
        print("  python test_endpoint.py <image_path_or_url>")
        print("\nExamples:")
        print("  python test_endpoint.py test.png")
        print("  python test_endpoint.py https://example.com/image.png")
        sys.exit(1)

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
