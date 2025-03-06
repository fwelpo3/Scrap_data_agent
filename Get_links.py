import asyncio
from crawl4ai import AsyncWebCrawler
from crawl4ai.async_configs import BrowserConfig, CrawlerRunConfig

async def _get_all_links_from_url_async(url: str) -> list[str]:
    """
    Asynchrone Funktion: Crawlt eine Webseite und gibt eine Liste aller Links zurück.
    (Diese Funktion bleibt im Wesentlichen gleich wie deine ursprüngliche asynchrone Funktion)
    """
    browser_config = BrowserConfig()
    run_config = CrawlerRunConfig()
    all_links = []
    async with AsyncWebCrawler(config=browser_config) as crawler:
        result = await crawler.arun(
            url=url,
            config=run_config
        )
        if result.success:
            if result.links:
                internal_links = result.links.get("internal", [])
                external_links = result.links.get("external", [])
                for link_data in internal_links:
                    all_links.append(link_data.get("href"))
                for link_data in external_links:
                    all_links.append(link_data.get("href"))
            else:
                print(f"Keine Links gefunden auf: {url}")
        else:
            print(f"Crawling fehlgeschlagen für: {url} - Fehler: {result.error_message}")
    return all_links

def get_links_from(url: str) -> list[str]:
    """
    Synchrone Funktion: Ruft die asynchrone Funktion auf und gibt die Links zurück.
    """
    return asyncio.run(_get_all_links_from_url_async(url)) # asyncio.run HIER VERWENDEN

# --- MAIN-Funktion und if __name__ Block ENTFERNT ---

# DIREKTER SYNCHRONER AUFRUF:
website_url = "https://www.tawil-media.de/"

links = get_links_from(website_url)

print(links)
if links:
    print(f"Alle Links gefunden auf {website_url}:")
    for link in links:
        print(f"- {link}")
