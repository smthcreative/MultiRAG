import json
from config import config

# --- Paths ---
RAW_ARTICLES = config.paths.RAW_ARTICLES
IMAGE_DESCRIPTIONS = config.paths.IMAGE_DESCRIPTIONS
MERGED_OUTPUT = config.paths.MERGED_ARTICLES



def load_data(raw=RAW_ARTICLES, desc=IMAGE_DESCRIPTIONS):
    if not raw.exists():
        print(f"ERROR: Input file not found: {raw}")
        exit(1)

    with open(raw, "r", encoding="utf-8") as f:
        articles = json.load(f)

    if not desc.exists():
        print(f"ERROR: Input file not found: {desc}")
        exit(1)
    with open(desc, "r", encoding="utf-8") as f:
        image_descriptions = json.load(f)

    return articles, image_descriptions



def merge(articles, image_descriptions):
    desc_map = {}
    for desc in image_descriptions:
        url = desc.get("article_url")
        if not url:
            continue
        desc_map.setdefault(url, []).append({
            "image_url": desc["original_image_url"],
            "description": desc["description"]
        })

    # merge articles with image descriptions, but only annotated images
    merged_articles = []
    for article in articles:
        url = article.get("url")
        if url in desc_map:
            merged_articles.append({
                "title": article.get("title", ""),
                "url": url,
                "text": article.get("text", ""),
                "images": desc_map[url]
            })
    return merged_articles

def save_mearged_articles(merged_articles):
    with open(MERGED_OUTPUT, "w", encoding="utf-8") as f:
        json.dump(merged_articles, f, ensure_ascii=False, indent=4)
    print(f"Saved {len(merged_articles)} articles with annotated images to {MERGED_OUTPUT}")

def main():
    articles, image_descriptions = load_data()
    merged_articles = merge(articles, image_descriptions)
    save_mearged_articles(merged_articles)

if __name__ == "__main__":
    main()



