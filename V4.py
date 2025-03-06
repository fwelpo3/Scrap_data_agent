import os
import json
import time
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass
from pathlib import Path
import asyncio

# Selenium Imports (bleiben gleich)
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager

# Gemini API (bleibt gleich)
import google.generativeai as genai

# Web Scraping (bleibt gleich)
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse

# Crawl4AI imports (bleibt gleich)
from crawl4ai import AsyncWebCrawler
from crawl4ai.async_configs import BrowserConfig, CrawlerRunConfig

# --- Logging-Konfiguration (bleibt gleich) ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('scraper.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# --- ScraperConfig (bleibt gleich) ---
@dataclass
class ScraperConfig:
    """Configuration settings for the scraper"""
    api_key: str
    model_name: str = "gemini-2.0-flash-thinking-exp-01-21"  # ggf aktuellstes Modell
    max_pages: int = 5
    delay_between_requests: float = 0
    max_content_length: int = 150000
    results_dir: str = "scraped_results"
    raw_content_dir: str = "raw_page_content"  # New directory for raw content
    temperature: float = 0.2
    top_p: float = 0.8
    top_k: int = 40
    max_output_tokens: int = 8192


class GeminiScraper:
    def __init__(self, config: ScraperConfig):
        """Initialize the Gemini Scraper with configuration"""
        self.config = config
        self.setup_gemini()
        self.browser = None
        self.visited_urls: Set[str] = set()
        # self.url_queue: List[str] = []  # Entfernt, da wir die Queue dynamisch verwalten.
        self.base_domain = ""

        # Create results and raw content directories
        Path(self.config.results_dir).mkdir(parents=True, exist_ok=True)
        Path(self.config.raw_content_dir).mkdir(parents=True, exist_ok=True)  # Create raw content dir

    def setup_gemini(self) -> None:
        """Configure the Gemini API"""
        genai.configure(api_key=self.config.api_key)
        self.models = [m for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
        self.model = genai.GenerativeModel(model_name=self.config.model_name)
        logger.info(f"Gemini model initialized: {self.model}")

        self.generation_config = {
            "temperature": self.config.temperature,
            "top_p": self.config.top_p,
            "top_k": self.config.top_k,
            "max_output_tokens": self.config.max_output_tokens,
        }

    def setup_browser(self) -> None:
        """Initialize Selenium browser with optimized settings"""
        if self.browser:
            return

        chrome_options = Options()
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--window-size=1920,1080")
        chrome_options.add_argument("--disable-extensions")
        chrome_options.add_argument("--disable-infobars")
        chrome_options.add_argument("--disable-notifications")
        chrome_options.add_argument("--disable-popup-blocking")

        service = Service(ChromeDriverManager().install())
        self.browser = webdriver.Chrome(service=service, options=chrome_options)
        logger.info("Browser initialized successfully")

    def __enter__(self):
        """Context manager entry"""
        self.setup_browser()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        if self.browser:
            self.browser.quit()

    def extract_domain(self, url: str) -> str:
        """Extract domain from URL"""
        parsed_url = urlparse(url)
        return f"{parsed_url.scheme}://{parsed_url.netloc}"

    async def _get_all_links_from_url_async(self, url: str) -> list[str]:
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

    def get_links_from(self, url: str) -> list[str]:
        """
        Synchrone Funktion: Ruft die asynchrone Funktion auf und gibt die Links zurück.
        """
        return asyncio.run(self._get_all_links_from_url_async(url))  # asyncio.run HIER VERWENDEN

    def get_page_content(self, url: str) -> str:
        """Get page content with optimized extraction"""
        try:
            self.browser.get(url)
            WebDriverWait(self.browser, 10).until(
                EC.presence_of_element_located((By.TAG_NAME, "body"))
            )

            soup = BeautifulSoup(self.browser.page_source, 'html.parser')

            # Remove unnecessary elements
            for element in soup(["script", "style", "meta", "noscript", "header", "footer", "nav"]):
                element.decompose()

            # Extract and clean text
            text = ' '.join(soup.stripped_strings)
            return ' '.join(text.split())  # Normalize whitespace

        except Exception as e:
            logger.error(f"Error getting content from {url}: {e}")
            return ""

    def analyze_content_with_gemini(self, content: str) -> Dict[str, Any]:
        """Analyze content with Gemini API"""

        schema = {
            "first_name": "",
            "last_name": "",
            "date_of_birth": "",
            "place_of_birth": "",
            "address": {
                "street": "",
                "house_number": "",
                "postal_code": "",
                "city": ""
            },
            "phone_number": "",
            "email": "",
            "description": ""
        }

        prompt = f"""
        Analyze the following website content and extract personal or company-related information.  
        Do not add extra text that does not fit into the specified fields. If you can predict missing data, include it.  
        If multiple people are detected, return them as an array.  

        The following fields should be identified:  

        Person1  
        - first_name  
        - last_name  
        - date_of_birth   
        - address (street, house_number, postal_code, city)  
        - phone_number  
        - email  
        - description (e.g., about the person or the company)  

        If more people are found (Person2, Person3, Person4, etc.), include them similarly.  

        Here is the content:  
        {content[:self.config.max_content_length]}
        """

        try:
            response = self.model.generate_content(
                prompt,
                generation_config=self.generation_config
            )

            try:
                response_text = response.text
                json_str = (
                    response_text.split("```json")[1].split("```")[0].strip()
                    if "```json" in response_text
                    else response_text.split("```")[1].split("```")[0].strip()
                    if "```" in response_text
                    else response_text
                )

                result = json.loads(json_str)
                return self._merge_with_default_schema(result, schema)

            except json.JSONDecodeError as e:
                logger.error(f"Error parsing JSON response: {e}")
                logger.error(f"Response text: {response.text}")
                return schema

        except Exception as e:
            logger.error(f"Error in analysis: {e}")
            return schema

    def prioritize_links_with_gemini(self, links: List[str], priority_instructions: str = None) -> Optional[List[str]]:
        """
        Priorisiert eine Liste von Links mithilfe des Gemini-Modells und behandelt Markdown-formatierte JSON-Antworten.
        """

        if not links:
            logging.warning("Die Liste der Links ist leer. Es wird nichts priorisiert.")
            return []

        default_instructions = "Bitte priorisiere die folgenden Links basierend auf ihrer Wahrscheinlichkeit, persönliche Informationen wie Namen, Adressen, E-Mail-Adressen, Telefonnummern oder berufliche Informationen zu enthalten. Sortiere die Links so, dass die wahrscheinlich relevantesten Links für die Suche nach persönlichen Informationen zuerst stehen. Antworte NUR mit einer JSON-Liste der sortierten Links."
        prompt_instructions = priority_instructions if priority_instructions else default_instructions

        prompt_content = f"""
        {prompt_instructions}

        Links:
        {json.dumps(links, ensure_ascii=False)}
        """

        try:
            response = self.model.generate_content(
                prompt_content,
                generation_config=self.generation_config
            )
            response.resolve()  # wait for response to be available

            if response.parts:
                response_text = response.text
                json_str = response_text  # Initialisiere json_str mit dem gesamten Antworttext

                # Versuche, JSON aus Markdown-Codeblöcken zu extrahieren
                if "```json" in response_text:
                    try:
                        json_str = response_text.split("```json")[1].split("```")[0].strip()
                    except IndexError:
                        logging.warning(
                            "Ungültige Markdown-JSON-Antwortstruktur erkannt, versuche es mit einfacher ``` Extraktion.")
                        try:  # Fallback für einfache ``` Codeblöcke (ohne json Kennzeichnung)
                            json_str = response_text.split("```")[1].split("```")[0].strip()
                        except IndexError:
                            logging.error("Kein JSON-Codeblock in der Gemini-Antwort gefunden.")
                            logging.error(f"Gemini Antworttext: {response_text}")
                            return None
                elif "```" in response_text:  # Fallback für einfache ``` Codeblöcke (ohne json Kennzeichnung)
                    try:
                        json_str = response_text.split("```")[1].split("```")[0].strip()
                    except IndexError:
                        logging.error("Ungültige einfache Markdown-Codeblock Struktur erkannt.")
                        logging.error(f"Gemini Antworttext: {response_text}")
                        return None

                try:
                    sorted_links_json = json.loads(json_str)
                    if isinstance(sorted_links_json, list):  # Stelle sicher, dass es eine Liste ist
                        logging.info("Links erfolgreich von Gemini priorisiert.")
                        return sorted_links_json
                    else:
                        logging.error(
                            f"Unerwartetes JSON-Format von Gemini erhalten. Erwartete Liste, bekam: {type(sorted_links_json)}")
                        logging.error(
                            f"Gemini Antworttext (extrahiert): {json_str}")  # Logge den extrahierten JSON-String
                        return None
                except json.JSONDecodeError as e:
                    logging.error(f"Fehler beim Parsen der JSON-Antwort von Gemini: {e}")
                    logging.error(f"Gemini Antworttext (extrahiert): {json_str}")  # Logge den extrahierten JSON-String
                    return None
            else:
                logging.error("Leere Antwort von Gemini erhalten.")
                return None


        except Exception as e:
            logging.error(f"Fehler bei der Kommunikation mit Gemini oder Verarbeitung der Antwort: {e}")
            return None

    def _merge_with_default_schema(self, result: Dict[str, Any], default_schema: Dict[str, Any]) -> Dict[str, Any]:
        """Merge result with default schema"""
        merged = default_schema.copy()

        for key, value in result.items():
            if key == "adresse" and isinstance(value, dict):
                merged["adresse"].update(value)
            elif key in merged and not merged[key] and value:
                merged[key] = value
            elif key not in merged:
                merged.setdefault("zusätzliche_infos", {})[key] = value

        return merged

    def crawl_website(self, start_url: str) -> Dict[str, Any]:
        """
        Crawls a website, starting from the given URL, and extracts information.
        This version prioritizes links *before* crawling.
        """
        self.base_domain = self.extract_domain(start_url)
        self.visited_urls.clear()
        all_page_contents = []  # List to store page contents

        # 1. Get *all* initial links from the start_url
        initial_links = self.get_links_from(start_url)
        initial_links.append(start_url)  # Füge die Start-URL selbst hinzu

        # 2. Prioritize the links using Gemini
        prioritized_links = self.prioritize_links_with_gemini(initial_links)

        if prioritized_links is None:
            logger.warning("Priorisierung fehlgeschlagen, verwende ursprüngliche Link-Liste.")
            prioritized_links = initial_links  # Fallback: ursprüngliche Liste

        # Limit the number of links to crawl, based on priority.
        links_to_crawl = prioritized_links[:self.config.max_pages]

        # 3. Crawl the prioritized links
        pages_crawled = 0
        for current_url in links_to_crawl:
            if current_url in self.visited_urls:
                continue

            logger.info(f"Crawling page {pages_crawled + 1}/{len(links_to_crawl)}: {current_url}")
            self.visited_urls.add(current_url)
            page_content = self.get_page_content(current_url)

            if page_content:
                all_page_contents.append(page_content)

            pages_crawled += 1
            time.sleep(self.config.delay_between_requests)

        logger.info(f"Crawling completed. {pages_crawled} pages processed.")

        # Save combined content to a single text file
        combined_content_filename = Path(self.config.raw_content_dir) / "combined_page_content.txt"
        combined_content = "\n\n".join(all_page_contents)  # Join pages with double newline
        with open(combined_content_filename, 'w', encoding='utf-8') as f:
            f.write(combined_content)
        logger.info(f"Combined page content saved to: {combined_content_filename}")

        if combined_content:
            final_extracted_info = self.analyze_content_with_gemini(combined_content)
            return final_extracted_info
        else:
            return {}  # leeres Dictionary zurück

    def save_results(self, results: Dict[str, Any], url: str) -> str:
        """Save results to JSON file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        domain = urlparse(url).netloc.replace(".", "_")
        filename = Path(self.config.results_dir) / f"{domain}_{timestamp}.json"

        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=4)

        logger.info(f"Results saved to: {filename}")
        return str(filename)


def main():
    """Main function"""
    config = ScraperConfig(api_key="AIzaSyC9N8ydc_4IdCybpJy5i3xgohCkKIJd6QE")  # Replace with your API key
    scraper = GeminiScraper(config)
    start_url = "https://www.tawil-media.de/"  # Example URL

    with scraper as s:
        results = s.crawl_website(start_url)
        if results:
            file_path = s.save_results(results, start_url)
            print(f"Crawling and analysis complete. Results exported to {file_path}")
        else:
            print("Crawling and analysis complete, but no data was extracted.")


if __name__ == "__main__":
    main()