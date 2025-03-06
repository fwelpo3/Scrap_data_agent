# Gemini Web Scraper - Version Evolution

Quick overview of Gemini Scraper versions and their advancements. Choose the version that fits your project's needs best!

## Version Highlights

| Version | Key Feature Focus         | Improvement Over Previous | Best For                                    | Trade-offs                                  |
|---------|---------------------------|---------------------------|---------------------------------------------|---------------------------------------------|
| **v1**    | Basic Crawl & Analyze     | Initial Implementation    | Simple tasks, learning the basics           | Inefficient, page-by-page analysis          |
| **v2**    | Combined Content Analysis | Holistic Website View     | Better context, improved accuracy          | Still slow, no priority crawling           |
| **v3**    | English Schema & Prompts  | Global Reach              | International projects, English websites    | Schema needs customization for specific data |
| **v4**    | Gemini Link Prioritization| Smart Crawling            | Targeted data, efficient resource use       | Relies on Gemini prioritization accuracy   |
| **v5**    | Async & Robust Crawling   | Performance & Scale       | Large websites, speed & reliability critical| Code complexity increases                  |

## Version Deep Dive

**v1: Core Functionality**

* Initial scraper. Crawls websites with Selenium and analyzes each page's content individually using Gemini.  Good for understanding the fundamental scraping process.

**v2: Holistic Analysis**

* Saves raw page content and analyzes the *combined* website content with Gemini. Provides Gemini with website-wide context for potentially better insights.

**v3: Internationalized**

* Schema and prompts updated to English for broader use.  More generic data schema for wider website compatibility.

**v4: Smart & Efficient**

* **Key Upgrade:** Gemini now *prioritizes links* *before* crawling, focusing on the most promising pages first for faster, targeted results.

**v5: Asynchronous & Production-Ready**

* **Major Leap:**  Asynchronous crawling with `asyncio` and `httpx` for massive speed gains.  Robust with error handling, `robots.txt` respect, and concurrent Selenium management. Built for scale and reliability.

## Version Selector

* **Start Simple:** `v1` or `v2` for basic tasks and learning.
* **English Projects:** `v3` for immediate use on English websites.
* **Targeted Scraping:** `v4` for efficient, relevant data extraction.
* **Performance & Scale:** `v5` for large, fast, and robust scraping needs.  More complex setup but highest performance.

**🔑 API Key Note:**  Remember to set your Gemini API key (`GEMINI_API_KEY` environment variable for v5, or directly in code for older versions).
