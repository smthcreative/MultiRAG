import json
import base64
import mimetypes
import time
from io import BytesIO
from pathlib import Path
import requests
from tqdm import tqdm


from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from config import config


# path
INPUT_JSON = config.paths.RAW_ARTICLES
OUTPUT_JSON = config.paths.IMAGE_DESCRIPTIONS
# rate limiter
REQUEST_DELAY_SECONDS = config.image_description.REQUEST_DELAY
MODEL_NAME = config.image_description.MODEL_NAME


def download_image_as_bytes(image_url: str):
    """Download image from URL and return BytesIO object"""
    try:
        resp = requests.get(image_url, timeout=10)
        resp.raise_for_status()
        return BytesIO(resp.content)
    except Exception as e:
        print(f"Failed to download image {image_url}: {e}")
        return None

def get_mime_type(image_url: str) -> str:
    """MIME type of image"""
    mime_type, _ = mimetypes.guess_type(image_url)
    return mime_type or "application/octet-stream"

def save_incremental_result(entry: dict, output_path: Path):
    """Append a single entry to JSON file safely"""
    data = []
    if output_path.exists():
        try:
            with open(output_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            pass
    data.append(entry)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def generate_image_descriptions(articles: list[dict], model_name=MODEL_NAME) -> list[dict]:
    """Generate image descriptions using Gemini API"""
    model = ChatGoogleGenerativeAI(model=model_name)
    generated_descriptions = []

    for article in tqdm(articles, desc="Processing articles"):
        title = article.get("title", "Untitled")
        url = article.get("url", "")
        image_urls = article.get("images", [])

        if not image_urls:
            continue

        for idx, img_url in enumerate(image_urls):
            print(f"\nProcessing image {idx+1}/{len(image_urls)} for article '{title}'")

            img_bytes = download_image_as_bytes(img_url)
            if not img_bytes:
                continue

            mime_type = get_mime_type(img_url)
            base64_str = base64.b64encode(img_bytes.getvalue()).decode("utf-8")

            message = HumanMessage(
                content=[
                    {
                        "type": "text",
                        "text": ("""
                        Describe only what is directly visible in the image. Follow these guidelines:

                        1. Decorative or non-informational images: respond with "<---image--->".
                        2. General images: list all visible objects, text, colors, shapes, positions, and measurable attributes. 
                           Do not write phrases like "Here is an image of" or similar introductions.
                        3. Charts, graphs, and infographics: include all labels, numbers, scales, and legends exactly as shown.
                        4. GIFs or sequential images: describe frame-by-frame changes or provide an accurate summary of the sequence.
                        
                        Rules:
                        * Only describe what is visible; do not interpret, infer, or provide analysis.
                        * Preserve all text, numbers, and symbols exactly as they appear.
                        * Be comprehensive, detailed, and precise, capturing every key visual element.
                        * Avoid any introductory or filler sentences; go straight to the description.
"""

                        )
                    },
                    {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{base64_str}"}}
                ]
            )

            try:
                response = model.invoke([message])
                desc_entry = {
                    "description": response.content,
                    "article_title": title,
                    "article_url": url,
                    "image_index": idx,
                    "original_image_url": img_url
                }

                # Save incrementally after each image as I use free tier(doesn`t have enough request to procced all images)
                save_incremental_result(desc_entry, OUTPUT_JSON)
                generated_descriptions.append(desc_entry)

            except Exception as e:
                print(f"Failed to generate description for image {img_url}: {e}")

            time.sleep(REQUEST_DELAY_SECONDS)

    return generated_descriptions

def main():
    config.validate()

    if not INPUT_JSON.exists():
        print(f"ERROR: Input file not found: {INPUT_JSON}")
        exit(1)

    with open(INPUT_JSON, "r", encoding="utf-8") as f:
        articles_data = json.load(f)

    articles_with_images = [a for a in articles_data if a.get("images")]

    if not articles_with_images:
        print("No articles with images found. Saving empty output.")
        OUTPUT_JSON.write_text("[]", encoding="utf-8")
        exit()

    print(f"Found {len(articles_with_images)} articles with images. Generating descriptions...")

    generate_image_descriptions(articles_with_images)

    print(f"Completed. Image descriptions saved incrementally to {OUTPUT_JSON}")

if __name__ == "__main__":
    main()
