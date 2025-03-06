import os
import json
import time
import logging
import argparse
from datetime import datetime
from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass
from pathlib import Path

# Selenium Imports
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager

# Gemini API
import google.generativeai as genai

# Web Scraping
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('scraper.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class ScraperConfig:
    """Configuration settings for the scraper"""
    api_key: str
    model_name: str = "gemini-2.0-flash-thinking-exp-01-21"
    max_pages: int = 10
    delay_between_requests: float = 2
    max_content_length: int = 150000
    results_dir: str = "scraped_results"
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
        self.url_queue: List[str] = []
        self.base_domain = ""
        
        # Create results directory
        Path(self.config.results_dir).mkdir(parents=True, exist_ok=True)

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

    def get_links_from_page(self, url: str) -> List[str]:
        """Extract all links from a page"""
        try:
            self.browser.get(url)
            WebDriverWait(self.browser, 10).until(
                EC.presence_of_element_located((By.TAG_NAME, "body"))
            )

            return [
                href for a_tag in self.browser.find_elements(By.TAG_NAME, "a")
                if (href := a_tag.get_attribute("href")) 
                and href.startswith(self.base_domain) 
                and href not in self.visited_urls
            ]
        except Exception as e:
            logger.error(f"Error extracting links from {url}: {e}")
            return []

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
        default_schema = {
            "vorname": "",
            "nachname": "",
            "geburtsdatum": "",
            "geburtsort": "",
            "geschlecht": "",
            "adresse": {
                "straße": "",
                "hausnummer": "",
                "postleitzahl": "",
                "ort": ""
            },
            "telefonnummer": "",
            "email": "",
            "beschreibung": ""
        }

        prompt = f"""
        Analysiere den folgenden Webseiteninhalt und extrahiere persönlichen oder unternehmensbezogenen Informationen.
        folgende Felder solltest du finden:
        - vorname
        - nachname
        - geburtsdatum
        - geburtsort
        - geschlecht
        - adresse (straße, hausnummer, postleitzahl, ort)
        - telefonnummer
        - email
        - beschreibung (z.B. der Person oder des Unternehmens)

        

        Hier ist der Inhalt:
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
                return self._merge_with_default_schema(result, default_schema)

            except json.JSONDecodeError as e:
                logger.error(f"Error parsing Gemini JSON response: {e}")
                logger.error(f"Response text: {response.text}")
                return default_schema

        except Exception as e:
            logger.error(f"Error in Gemini analysis: {e}")
            return default_schema

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
        """Crawl website and extract information"""
        self.base_domain = self.extract_domain(start_url)
        self.url_queue = [start_url]
        self.visited_urls.clear()

        all_extracted_info = {
            "vorname": "",
            "nachname": "",
            "geburtsdatum": "",
            "geburtsort": "",
            "geschlecht": "",
            "staatsangehörigkeit": "",
            "familienstand": "",
            "adresse": {
                "straße": "",
                "hausnummer": "",
                "postleitzahl": "",
                "ort": ""
            },
            "telefonnummer": "",
            "email": "",
            "steueridentifikationsnummer": "",
            "beschreibung": "",
            "zusätzliche_infos": {}
        }

        pages_crawled = 0

        while self.url_queue and pages_crawled < self.config.max_pages:
            current_url = self.url_queue.pop(0)
            if current_url in self.visited_urls:
                continue

            logger.info(f"Crawling page {pages_crawled + 1}/{self.config.max_pages}: {current_url}")
            self.visited_urls.add(current_url)

            page_content = self.get_page_content(current_url)
            if page_content:
                extracted_info = self.analyze_content_with_gemini(page_content)
                self._merge_info(all_extracted_info, extracted_info)

                new_links = self.get_links_from_page(current_url)
                self.url_queue.extend(link for link in new_links if link not in self.visited_urls)

            pages_crawled += 1
            logger.info(f"Progress: {pages_crawled}/{self.config.max_pages} pages processed")
            time.sleep(self.config.delay_between_requests)

        logger.info(f"Crawling completed. {pages_crawled} pages processed.")
        return all_extracted_info

    def _merge_info(self, target: Dict[str, Any], source: Dict[str, Any]) -> None:
        """Merge information from different pages"""
        for key, value in source.items():
            if key == "adresse" and isinstance(value, dict) and isinstance(target.get(key), dict):
                target[key].update(value)
            elif key in target and not target[key] and value:
                target[key] = value
            elif key not in target:
                target.setdefault("zusätzliche_infos", {})[key] = value

    def save_results(self, results: Dict[str, Any], url: str) -> str:
        """Save results to JSON file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        domain = urlparse(url).netloc.replace(".", "_")
        filename = Path(self.config.results_dir) / f"{domain}_{timestamp}.json"

        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=4)

        logger.info(f"Results saved to: {filename}")
        return str(filename)

class ScrapeBot:
    def __init__(self, config: ScraperConfig):
        """Initialize the ScrapeBot"""
        self.config = config
        self.scraper = GeminiScraper(config)
        self.history: List[Dict[str, str]] = []
        self.welcome_message = """
        ╔══════════════════════════════════════════════════╗
        ║                  GEMINI SCRAPER                  ║
        ╚══════════════════════════════════════════════════╝

        Willkommen beim Gemini Scraper!
        Dieser Bot hilft dir, Webseiten nach persönlichen und
        unternehmensbezogenen Informationen zu durchsuchen.

        Befehle:
        - 'scrape <url>': Crawlt eine Webseite und extrahiert Informationen
        - 'info': Zeigt Informationen zum aktuellen Status
        - 'history': Zeigt den Chatverlauf
        - 'exit' oder 'quit': Beendet das Programm

        Starte mit dem Befehl 'scrape <url>', um loszulegen.
        """

    def add_to_history(self, role: str, message: str) -> None:
        """Add message to chat history"""
        self.history.append({
            "role": role,
            "message": message,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })

    def show_history(self) -> None:
        """Display chat history"""
        if not self.history:
            print("Chat history is empty.")
            return

        print("\n=== CHAT HISTORY ===")
        for entry in self.history:
            print(f"[{entry['timestamp']}] {entry['role']}: {entry['message'][:100]}...")
        print("==================\n")

    def process_command(self, command: str) -> bool:
        """Process user command"""
        self.add_to_history("user", command)
        command_lower = command.lower().strip()

        if command_lower.startswith('scrape '):
            url = command[7:].strip()
            response = f"Starting crawl for: {url}"
            self.add_to_history("bot", response)
            logger.info(response)

            try:
                with self.scraper as scraper:
                    results = scraper.crawl_website(url)
                    file_path = scraper.save_results(results, url)

                    print("\n=== EXTRACTED INFORMATION ===")
                    print(json.dumps(results, ensure_ascii=False, indent=2))
                    print(f"Results exported to {file_path}")

                    response = f"Crawling completed. Results exported to {file_path}"
                    self.add_to_history("bot", response)
            except Exception as e:
                response = f"Error during crawling: {e}"
                self.add_to_history("bot", response)
                logger.error(response)
                return True

        elif command_lower in ['exit', 'quit']:
            return False
        elif command_lower == 'history':
            self.show_history()
        elif command_lower == 'info':
            print(f"Current configuration:\n{self.config}")
        else:
            print("Unknown command. Type 'help' for available commands.")

        return True

    def run(self) -> None:
        """Start the interactive bot"""
        print(self.welcome_message)

        try:
            while True:
                user_input = input("\n> ")
                if not self.process_command(user_input):
                    break
        except KeyboardInterrupt:
            print("\nShutting down...")
        except Exception as e:
            logger.error(f"Error: {e}")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Gemini Website Scraper')
    parser.add_argument('--url', help='URL to scrape')
    args = parser.parse_args()

    config = ScraperConfig(
        api_key="AIzaSyC9N8ydc_4IdCybpJy5i3xgohCkKIJd6QE"
    )
    
    bot = ScrapeBot(config)

    if args.url:
        bot.process_command(f"scrape {args.url}")
        return

    bot.run()

if __name__ == "__main__":
    main()