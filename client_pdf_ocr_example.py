#!/usr/bin/env python3
"""
Example client script showing correct PDF OCR usage with the simplified RunPod handler.

This demonstrates the recommended architecture:
1. Client-side: Convert PDF to images using Chandra's native load_pdf_images()
2. Server-side: RunPod handler processes only images (not PDFs)

Benefits:
- Leverages Chandra's battle-tested PDF processing
- Eliminates serverless file system complexity
- Client has full control over page ranges
- Reduced handler complexity = fewer error points
"""

import requests
import base64
import json
import time
import os
from dotenv import load_dotenv
from chandra.input import load_pdf_images  # Use original Chandra's PDF processor

# Load environment variables
load_dotenv()

ENDPOINT_ID = os.getenv("RUNPOD_ENDPOINT_ID")
API_KEY = os.getenv("RUNPOD_API_KEY")
PDF_FILE = "202510-104_Jaron.pdf"  # 9-page PDF example

print("=" * 80)
print("📄 Chandra PDF OCR - Client-Side Conversion Example")
print("=" * 80)
print(f"\n✅ Endpoint: {ENDPOINT_ID}")
print(f"✅ PDF File: {PDF_FILE}")
print(f"✅ Architecture: Client-side PDF→Images, Server-side OCR\n")

# Step 1: Convert PDF to images CLIENT-SIDE using Chandra's native function
print("=" * 80)
print("STEP 1: Convert PDF to Images (Client-Side)")
print("=" * 80)
print(f"\n📖 Loading PDF: {PDF_FILE}")

# Use Chandra's original load_pdf_images() function
# This is the CORRECT way - leverages existing, tested PDF processing
images = load_pdf_images(PDF_FILE, page_range=None)  # None = all pages

print(f"✅ PDF converted to {len(images)} images")
for i, img in enumerate(images):
    print(f"   Page {i+1}: {img.size[0]}x{img.size[1]} pixels")

# Step 2: Convert images to base64
print(f"\n📦 Converting images to base64...")
images_base64 = []
for i, pil_image in enumerate(images):
    from io import BytesIO
    buffer = BytesIO()
    pil_image.save(buffer, format="PNG")
    img_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
    images_base64.append(img_base64)
    size_kb = len(img_base64) / 1024
    print(f"   Page {i+1}: {size_kb:.2f} KB (base64)")

total_size_mb = sum(len(b) for b in images_base64) / 1024 / 1024
print(f"\n✅ Total payload size: {total_size_mb:.2f} MB\n")

# Step 3: Send to RunPod OCR endpoint
print("=" * 80)
print("STEP 2: Send Images to RunPod for OCR")
print("=" * 80)
print(f"\n📤 Sending {len(images_base64)} images to RunPod endpoint...")

url = f"https://api.runpod.ai/v2/{ENDPOINT_ID}/run"
headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {API_KEY}"
}

payload = {
    "input": {
        "images": images_base64,  # Send multiple images
        "max_output_tokens": 12384,
        "include_images": True,
        "include_headers_footers": False,
        "prompt_type": "ocr_layout"
    }
}

response = requests.post(url, headers=headers, json=payload, timeout=30)
result = response.json()

job_id = result.get("id")
print(f"✅ Job ID: {job_id}")
print(f"✅ Status: {result.get('status')}\n")

# Step 4: Monitor processing status
print("=" * 80)
print("STEP 3: Monitor OCR Processing")
print("=" * 80)
print()

start_time = time.time()
status_url = f"https://api.runpod.ai/v2/{ENDPOINT_ID}/status/{job_id}"

for iteration in range(150):  # Max 5 minutes
    elapsed = int(time.time() - start_time)

    # Query status
    response = requests.get(status_url, headers=headers, timeout=10)
    result = response.json()
    status = result.get("status")

    # Display status
    if iteration % 5 == 0:  # Every 10 seconds
        print(f"[{elapsed:3d}s] {status}")

    if status == "COMPLETED":
        print(f"\n{'='*80}")
        print(f"🎉 Processing Complete! ({elapsed} seconds)")
        print(f"{'='*80}\n")

        # Display results
        output = result.get("output", {})
        pages = output.get("total_pages", 0)
        tokens = output.get("total_tokens", 0)

        print("📊 Results:")
        print(f"   - Total Pages: {pages}")
        print(f"   - Total Tokens: {tokens}")
        print(f"   - Average/Page: {tokens // pages if pages else 0} tokens\n")

        # Save results
        with open("ocr_result_clientside_pdf.json", "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
        print("✅ JSON saved: ocr_result_clientside_pdf.json")

        all_markdown = "\n\n---\n\n".join([
            f"# Page {i+1}\n\n{r['markdown']}"
            for i, r in enumerate(output["results"])
        ])
        with open("ocr_result_clientside_pdf.txt", "w", encoding="utf-8") as f:
            f.write(all_markdown)
        print("✅ Markdown saved: ocr_result_clientside_pdf.txt")

        # Preview first page
        if output.get("results"):
            first_page = output["results"][0]["markdown"]
            print(f"\n{'='*80}")
            print("📄 Page 1 Preview (first 600 chars):")
            print(f"{'='*80}\n")
            print(first_page[:600])
            if len(first_page) > 600:
                print(f"\n... ({len(first_page) - 600} more characters)")
            print(f"\n{'='*80}")

        break

    elif status == "FAILED":
        print(f"\n❌ Failed!")
        print(json.dumps(result, indent=2, ensure_ascii=False))
        break

    elif status == "IN_PROGRESS":
        if iteration % 5 == 0:
            print("      → Worker processing images...")

    elif status == "IN_QUEUE":
        if iteration % 5 == 0:
            print("      → Waiting for worker...")

    # Timeout check
    if elapsed > 300:  # 5 minutes
        print(f"\n⏱️ Timeout (5 minutes)")
        break

    # Wait 2 seconds
    time.sleep(2)

print("\n✅ Example completed!")
print("\n" + "=" * 80)
print("ARCHITECTURE SUMMARY")
print("=" * 80)
print("""
This script demonstrates the CORRECT architecture:

CLIENT-SIDE (this script):
  1. Use chandra.input.load_pdf_images() to convert PDF to PIL Images
  2. Convert PIL Images to base64
  3. Send base64 images to RunPod endpoint

SERVER-SIDE (RunPod handler):
  1. Decode base64 back to PIL Images
  2. Run OCR inference on each image
  3. Return results in BatchOutputItem format

BENEFITS:
  ✓ Leverages Chandra's battle-tested PDF processing
  ✓ No file system operations in serverless environment
  ✓ Client controls page ranges and conversion settings
  ✓ Simpler handler = fewer bugs
  ✓ Follows Single Responsibility Principle
""")
