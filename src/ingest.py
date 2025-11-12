import requests
import time
import json
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from pathlib import Path
from config import config


BASE_URL = config.scraping.BASE_URL
OUT_PATH = config.paths.RAW_ARTICLES
HEADERS = config.scraping.HEADERS
PAGES = config.scraping.NUM_PAGES
DELAY = config.scraping.REQUEST_DELAY


def get_article_cards(page_url):
    """Extract article cards (title, link, date, main image) from a Batch listing page"""
    res = requests.get(page_url, headers=HEADERS)
    res.raise_for_status()
    soup = BeautifulSoup(res.text, "html.parser")

    articles = []
    for article in soup.find_all("article"):
        # check if the article not in PostCardSmall(as they contain popular articles)
        if article.get('data-sentry-component') == 'PostCardSmall':
            continue

        # title
        title_tag = article.find("h2")
        title = title_tag.get_text(strip=True) if title_tag else None

        # link and date
        date_tag, link_tag = article.find_all("a", href=True)
        if not link_tag:
            continue

        link = urljoin(BASE_URL, link_tag["href"])

        date = date_tag.get_text(strip=True) if date_tag else None

        articles.append({
            "title": title,
            "url": link,
            "date": date,
        })
    return articles

def get_article_content(article_url):
    """Extract text and images"""
    res = requests.get(article_url, headers=HEADERS)
    res.raise_for_status()
    soup = BeautifulSoup(res.text, "html.parser")

    # extract article body
    content_div = soup.find("article")
    if not content_div:
        content_div = soup  # fallback

    paragraphs = [p.get_text(" ", strip=True) for p in content_div.select("p, h2, li")]
    text = "\n".join(paragraphs)

    # extract all images
    images = []
    for img in content_div.find_all("img"):
        if img.get("src") and not img["src"].startswith("data:"):
            if '_next/image' in img["src"]:
                continue
            images.append(urljoin(BASE_URL, img["src"]))

    return {
        "text": text,
        "images": list(set(images))
    }


def scrape_batch_articles(pages=3, delay=1.5, verbose=False):
    """Scrape all articles from N pages of The Batch"""
    all_articles = []
    for i in range(1, pages + 1):
        page_url = f"{BASE_URL}/page/{i}"
        cards = get_article_cards(page_url)
        if verbose:
            print(f"Scraping listing page {i}: {page_url}")
            print(f"Found {len(cards)} articles on page {i}")
        for card in cards:
            if verbose:
                print(f"  ↳ Fetching: {card['title']}")
            details = get_article_content(card["url"])
            card.update(details)
            all_articles.append(card)
            time.sleep(delay)

    return all_articles


def save_articles(articles: list[dict], output_path: Path = None):
    """Save scraped articles to JSON file"""

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(articles, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(articles)} articles to {output_path}")

def main():
    config.validate()
    articles = scrape_batch_articles(pages=PAGES, delay=DELAY, verbose=True)
    save_articles(articles, OUT_PATH)

if __name__ == "__main__":
    main()



