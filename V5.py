import asyncio
import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
from urllib.parse import urlparse, urljoin
from urllib.robotparser import RobotFileParser

import aiofiles
import google.generativeai as genai
import httpx
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from webdriver_manager.chrome import ChromeDriverManager

# --- Konstanten ---
DEFAULT_PRIORITY_INSTRUCTIONS = (
    "Please prioritize the following links based on their likelihood of containing "
    "personal or company-related information (names, addresses, emails, phone numbers, etc.). "
    "Return a JSON list of the sorted links, with the most relevant links first."
)

DEFAULT_PERSON_SCHEMA = {
    "first_name": "",
    "last_name": "",
    "date_of_birth": "",
    "place_of_birth": "",
    "address": {"street": "", "house_number": "", "postal_code": "", "city": ""},
    "phone_number": "",
    "email": "",
    "description": "",
}

MAX_RETRIES = 3
RETRY_DELAY = 2         # Sekunden
GEMINI_TIMEOUT = 30       # Sekunden
MAX_CONCURRENT_SELENIUM = 3  # Maximale gleichzeitige Selenium-Instanzen

# Logging-Konfiguration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("scraper.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


# --- Wiederverwendbare Retry-Logik ---
async def async_retry(func, *args, retries=MAX_RETRIES, delay=RETRY_DELAY, fallback=None, **kwargs):
    for attempt in range(retries):
        try:
            return await func(*args, **kwargs)
        except (TimeoutError, asyncio.TimeoutError) as e:
            logger.warning(f"{func.__name__} timed out (attempt {attempt + 1}/{retries}).")
        except Exception as e:
            logger.error(f"Error in {func.__name__} (attempt {attempt + 1}/{retries}): {e}")
        if attempt < retries - 1:
            await asyncio.sleep(delay * (2 ** attempt))
    return fallback


# --- Scraper-Konfiguration ---
@dataclass
class ScraperConfig:
    api_key: str
    model_name: str = "gemini-2.0-flash-thinking-exp-01-21"  # Oder dein bevorzugtes Modell
    max_pages: int = 5
    delay_between_requests: float = 0.5  # Kleine Standardverzögerung
    max_content_length: int = 150000
    results_dir: str = "scraped_results"
    raw_content_dir: str = "raw_page_content"
    temperature: float = 0.2
    top_p: float = 0.8
    top_k: int = 40
    max_output_tokens: int = 8192
    selenium_options: Optional[Options] = None  # Erlaubt benutzerdefinierte Selenium-Optionen
    httpx_options: Optional[Dict[str, Any]] = None  # Erlaubt benutzerdefinierte httpx-Optionen


# --- GeminiScraper-Klasse ---
class GeminiScraper:
    def __init__(self, config: ScraperConfig):
        self.config = config
        self.setup_gemini()
        self.browser: Optional[webdriver.Chrome] = None
        self.visited_urls: Set[str] = set()
        self.base_domain: str = ""
        self.semaphore = asyncio.Semaphore(MAX_CONCURRENT_SELENIUM)
        self.robot_parser: Optional[RobotFileParser] = None

        # Persistent httpx AsyncClient
        client_options = self.config.httpx_options if self.config.httpx_options else {}
        self.http_client = httpx.AsyncClient(
            headers={
                'User-Agent': ('Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                                 'AppleWebKit/537.36 (KHTML, like Gecko) '
                                 'Chrome/91.0.4472.124 Safari/537.36')
            },
            follow_redirects=True,
            timeout=10,
            **client_options
        )

        # Erstelle benötigte Verzeichnisse
        Path(self.config.results_dir).mkdir(parents=True, exist_ok=True)
        Path(self.config.raw_content_dir).mkdir(parents=True, exist_ok=True)

    def setup_gemini(self):
        genai.configure(api_key=self.config.api_key)
        self.model = genai.GenerativeModel(model_name=self.config.model_name)
        logger.info(f"Gemini model initialized: {self.config.model_name}")
        self.generation_config = genai.GenerationConfig(
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            top_k=self.config.top_k,
            max_output_tokens=self.config.max_output_tokens,
        )

    async def setup_browser(self):
        if self.browser:
            return

        async with self.semaphore:
            chrome_options = self.config.selenium_options or Options()
            # Standardmäßig im Headless-Modus
            if not any("--headless" in arg for arg in chrome_options.arguments):
                chrome_options.add_argument("--headless")
            chrome_options.add_argument("--no-sandbox")
            chrome_options.add_argument("--disable-dev-shm-usage")
            chrome_options.add_argument("--disable-gpu")
            chrome_options.add_argument("--window-size=1920,1080")

            service = Service(ChromeDriverManager().install())
            # Initialisiere den Browser in einem separaten Thread, um den Event-Loop nicht zu blockieren
            self.browser = await asyncio.to_thread(webdriver.Chrome, service=service, options=chrome_options)
            logger.info("Browser initialized successfully")

    async def __aenter__(self):
        await self.setup_browser()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.browser:
            await asyncio.to_thread(self.browser.quit)
        await self.http_client.aclose()

    def _extract_domain(self, url: str) -> str:
        parsed_url = urlparse(url)
        return f"{parsed_url.scheme}://{parsed_url.netloc}"

    def _init_robot_parser(self, url: str):
        domain = self._extract_domain(url)
        self.robot_parser = RobotFileParser()
        self.robot_parser.set_url(f"{domain}/robots.txt")
        try:
            self.robot_parser.read()
        except Exception as e:
            logger.warning(f"Error reading robots.txt: {e}")
            self.robot_parser = None

    def _can_fetch(self, url: str) -> bool:
        if self.robot_parser:
            return self.robot_parser.can_fetch("*", url)
        return True

    async def _fetch_with_httpx(self, url: str) -> Tuple[Optional[str], bool]:
        response = await self.http_client.get(url)
        response.raise_for_status()
        content_type = response.headers.get("content-type", "")
        is_html = "text/html" in content_type
        return response.text, is_html

    async def _get_all_links_from_url(self, url: str) -> List[str]:
        if not self._can_fetch(url):
            logger.info(f"Skipping (robots.txt): {url}")
            return []

        try:
            content, is_html = await self._fetch_with_httpx(url)
            if is_html and content:
                soup = BeautifulSoup(content, 'html.parser')
                links = [
                    self._normalize_link(url, a['href'])
                    for a in soup.find_all('a', href=True)
                    if self._is_valid_link(url, a['href'])
                ]
                return links
            # Fallback auf Selenium, falls httpx nicht HTML liefert
            logger.info(f"Using Selenium for: {url}")
            links = await self._get_links_with_selenium(url)
            return links
        except httpx.RequestError as e:
            logger.error(f"httpx request failed for {url}: {e}")
            return []
        except Exception as e:
            logger.error(f"Error getting links from {url}: {e}")
            return []

    async def _get_links_with_selenium(self, url: str) -> List[str]:
        if not self.browser:
            await self.setup_browser()
        async with self.semaphore:
            try:
                # Blockierende Selenium-Aufrufe in einem Thread ausführen
                await asyncio.to_thread(self.browser.get, url)
                await asyncio.to_thread(
                    WebDriverWait(self.browser, 10).until,
                    EC.presence_of_element_located((By.TAG_NAME, "body"))
                )
                soup = BeautifulSoup(self.browser.page_source, "html.parser")
                links = [
                    self._normalize_link(url, a["href"])
                    for a in soup.find_all("a", href=True)
                    if self._is_valid_link(url, a["href"])
                ]
                return links
            except TimeoutException:
                logger.warning(f"Selenium timed out for {url}")
                return []
            except Exception as e:
                logger.error(f"Selenium error for {url}: {e}")
                return []

    def _normalize_link(self, base_url: str, link: str) -> str:
        return urljoin(base_url, link)

    def _is_valid_link(self, base_url: str, link: str) -> bool:
        normalized_link = self._normalize_link(base_url, link)
        if not normalized_link.startswith(self.base_domain):
            return False
        if normalized_link in self.visited_urls:
            return False
        return True

    async def _fetch_with_selenium(self, url: str) -> Optional[str]:
        if not self.browser:
            await self.setup_browser()
        async with self.semaphore:
            try:
                await asyncio.to_thread(self.browser.get, url)
                await asyncio.to_thread(
                    WebDriverWait(self.browser, 10).until,
                    EC.presence_of_element_located((By.TAG_NAME, "body"))
                )
                return self.browser.page_source
            except TimeoutException:
                logger.warning(f"Selenium timed out for {url}")
                return None
            except Exception as e:
                logger.error(f"Selenium error for {url}: {e}")
                return None

    async def get_page_content(self, url: str) -> str:
        if not self._can_fetch(url):
            logger.info(f"Skipping (robots.txt): {url}")
            return ""
        try:
            content, is_html = await self._fetch_with_httpx(url)
            if is_html and content:
                return self._clean_html(content)
            logger.info(f"Using Selenium for (httpx failed or not HTML): {url}")
            page_source = await self._fetch_with_selenium(url)
            if page_source:
                return self._clean_html(page_source)
            else:
                return ""
        except httpx.RequestError as e:
            logger.error(f"httpx request failed for {url}: {e}")
            return ""
        except Exception as e:
            logger.error(f"Error getting content from {url}: {e}")
            return ""

    def _clean_html(self, html_content: str) -> str:
        soup = BeautifulSoup(html_content, "html.parser")
        for element in soup(["script", "style", "meta", "noscript", "header", "footer", "nav"]):
            element.decompose()
        text = " ".join(soup.stripped_strings)
        return " ".join(text.split())

    def _extract_json_from_markdown(self, response_text: str) -> Any:
        try:
            if "```json" in response_text:
                json_str = response_text.split("```json")[1].split("```")[0].strip()
                return json.loads(json_str)
            elif "```" in response_text:
                json_str = response_text.split("```")[1].split("```")[0].strip()
                return json.loads(json_str)
            else:
                logger.warning("No JSON code block found in response.")
                return json.loads(response_text)
        except (IndexError, json.JSONDecodeError) as e:
            logger.error(f"Failed to extract JSON: {e}")
            logger.error(f"Response Text:\n{response_text}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error during JSON extraction: {e}")
            raise

    def _build_analysis_prompt(self, content: str) -> str:
        return (
            "Analyze the following website content and extract personal or company-related information.\n"
            "Do not add extra text that does not fit into the specified fields. If you can predict missing data, include it.\n"
            "If multiple people are detected, return them as an array (Person1, Person2, Person3, ... with the same fields).\n\n"
            "The following fields should be identified:\n"
            " - first_name\n - last_name\n - date_of_birth\n - address (street, house_number, postal_code, city)\n"
            " - phone_number\n - email\n - description\n\n"
            "Here is the content:\n" + content[:self.config.max_content_length]
        )

    async def _analyze_content_with_gemini(self, content: str) -> Dict[str, Any]:
        prompt = self._build_analysis_prompt(content)

        async def gemini_call():
            response = await self.model.generate_content_async(
                prompt, generation_config=self.generation_config
            )
            parsed_json = self._extract_json_from_markdown(response.text)
            return self._merge_with_default_schema(parsed_json, DEFAULT_PERSON_SCHEMA)

        result = await async_retry(gemini_call, fallback=DEFAULT_PERSON_SCHEMA)
        if result == DEFAULT_PERSON_SCHEMA:
            logger.error("Gemini API analysis failed after multiple retries.")
        return result

    async def _prioritize_links_with_gemini(self, links: List[str],
                                              priority_instructions: Optional[str] = None) -> List[str]:
        if not links:
            return []

        prompt_instructions = priority_instructions or DEFAULT_PRIORITY_INSTRUCTIONS
        prompt_content = f"{prompt_instructions}\n\nLinks:\n{json.dumps(links, ensure_ascii=False)}"

        async def gemini_prioritize():
            response = await self.model.generate_content_async(
                prompt_content, generation_config=self.generation_config
            )
            parsed_json = self._extract_json_from_markdown(response.text)
            if isinstance(parsed_json, list):
                return parsed_json
            else:
                raise ValueError(f"Unexpected JSON format. Expected list, got {type(parsed_json)}")

        result = await async_retry(gemini_prioritize, fallback=links)
        if result == links:
            logger.warning("Gemini link prioritization failed; returning original links.")
        else:
            logger.info("Links successfully prioritized by Gemini.")
        return result

    def _merge_with_default_schema(self, result: Dict[str, Any], default_schema: Dict[str, Any]) -> Dict[str, Any]:
        merged = default_schema.copy()
        for key, value in result.items():
            if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
                merged[key] = self._merge_with_default_schema(value, merged[key])
            elif key in merged:
                merged[key] = value
            else:
                merged.setdefault("additional_info", {})[key] = value
        return merged

    async def _collect_page_content(self, links_to_crawl: List[str]) -> str:
        tasks = []
        pages_crawled = 0

        for current_url in links_to_crawl:
            if current_url in self.visited_urls:
                continue
            if pages_crawled >= self.config.max_pages:
                break
            self.visited_urls.add(current_url)
            logger.info(f"Scheduling crawl ({pages_crawled + 1}/{self.config.max_pages}): {current_url}")
            tasks.append(self.get_page_content(current_url))
            pages_crawled += 1

        results = await asyncio.gather(*tasks)
        combined_content = "\n\n".join(content for content in results if content)
        # Asynchrone Speicherung des kombinierten Inhalts
        combined_content_filename = Path(self.config.raw_content_dir) / "combined_page_content.txt"
        async with aiofiles.open(combined_content_filename, "w", encoding="utf-8") as f:
            await f.write(combined_content)
        logger.info(f"Combined page content saved to: {combined_content_filename}")
        return combined_content

    async def crawl_website(self, start_url: str) -> Dict[str, Any]:
        self.base_domain = self._extract_domain(start_url)
        self._init_robot_parser(start_url)
        self.visited_urls.clear()

        initial_links = await self._get_all_links_from_url(start_url)
        # Stelle sicher, dass auch die Start-URL verarbeitet wird
        if start_url not in initial_links:
            initial_links.append(start_url)

        prioritized_links = await self._prioritize_links_with_gemini(initial_links)
        combined_content = await self._collect_page_content(prioritized_links)

        if combined_content:
            final_extracted_info = await self._analyze_content_with_gemini(combined_content)
            return final_extracted_info
        else:
            return {}

    async def save_results(self, results: Dict[str, Any], url: str) -> str:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        domain = urlparse(url).netloc.replace(".", "_")
        filename = Path(self.config.results_dir) / f"{domain}_{timestamp}.json"
        async with aiofiles.open(filename, "w", encoding="utf-8") as f:
            await f.write(json.dumps(results, ensure_ascii=False, indent=4))
        logger.info(f"Results saved to: {filename}")
        return str(filename)


# --- Main-Funktion ---
async def main():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("Error: GEMINI_API_KEY environment variable not set.")
        return

    config = ScraperConfig(api_key=api_key)
    async with GeminiScraper(config) as scraper:
        start_url = "https://www.tawil-media.de/"  # Beispiel-URL
        results = await scraper.crawl_website(start_url)
        if results:
            file_path = await scraper.save_results(results, start_url)
            print(f"Crawling and analysis complete. Results exported to {file_path}")
        else:
            print("Crawling and analysis complete, but no data was extracted.")


if __name__ == "__main__":
    asyncio.run(main())
