import os
import requests
from bs4 import BeautifulSoup

# Custom headers to mimic a real browser
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/115.0.0.0 Safari/537.36"
}

# List the pages you want to scrape
urls = {
    "home": "https://cxneo.com/",
    "services": "https://cxneo.com/services/",
    "digital_marketing": "https://cxneo.com/services/ai-driven-digital-marketing-solutions/",
    "web_dev": "https://cxneo.com/services/website-design-development/",
    "automation": "https://cxneo.com/services/enterprise-marketing-automation-solutions/",
    "company": "https://cxneo.com/company/",
    "contact": "https://cxneo.com/contact/"
}

os.makedirs("content", exist_ok=True)

for name, url in urls.items():
    try:
        resp = requests.get(url, headers=headers, timeout=10)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

        # Extract visible text
        text = soup.get_text(separator="\n", strip=True)
        filepath = os.path.join("content", f"{name}.txt")
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(text)

        print(f"[✓] Saved {name}.txt")
    except Exception as e:
        print(f"[X] Failed to fetch {url}: {e}")