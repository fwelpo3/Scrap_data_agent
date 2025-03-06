# 🚀 Gemini Web Scraper - Version Evolution 🌟

_Your quick guide to choosing the right Gemini Scraper version for your project!_

<hr>

## ✨ Version Highlights ✨

| Version | **Key Feature Focus**         | ⚡ **Improvement** ⚡          | ✅ **Best For** ✅                             | 🤔 **Trade-offs** 🤔                               |
|---------|---------------------------|---------------------------|---------------------------------------------|---------------------------------------------|
| **v1**    | 🏗️ Basic Crawl & Analyze     | _Initial Implementation_    | 👶 Simple tasks, learning the basics           | 🐌 Inefficient, page-by-page analysis          |
| **v2**    | 📚 Combined Content Analysis | 🌐 _Holistic Website View_     | 🧠 Better context, improved accuracy          | 🐢 Still slow, no priority crawling           |
| **v3**    | 🌍 English Schema & Prompts  | 🗣️ _Global Reach_              | 🌎 International projects, English websites    | 🛠️ Schema needs customization for specific data |
| **v4**    | 🎯 Gemini Link Prioritization| 🧠 _Smart Crawling_            | 🎯 Targeted data, efficient resource use       | 🤖 Relies on Gemini prioritization accuracy   |
| **v5**    | ⚙️ Async & Robust Crawling   | 💪 _Performance & Scale_       | 🚀 Large websites, speed & reliability critical| 📈 Code complexity increases                  |

<hr>

## 🔎 Version Deep Dive 🔍

### <ins>**v1: Core Functionality**</ins> 🏗️

> The foundational scraper. Crawls websites with Selenium and analyzes each page's content individually using Gemini.  Perfect for understanding the basic scraping workflow.

### <ins>**v2: Holistic Analysis**</ins> 📚

> Saves raw page content and uses Gemini to analyze the *combined* website content.  Provides Gemini with a website-wide perspective for potentially deeper insights and better data quality.

### <ins>**v3: Internationalized**</ins> 🌍

> Schema and prompts are translated to English for broader applicability. Features a more generic data schema to enhance compatibility with diverse websites.

### <ins>**v4: Smart & Efficient**</ins> 🎯

> **🔥 Key Upgrade:** Gemini now *prioritizes links* *before* crawling! This intelligent approach focuses on the most promising pages, delivering faster, more targeted results and saving resources.

### <ins>**v5: Asynchronous & Production-Ready**</ins> ⚙️

> **🚀 Major Leap:** Achieves massive speed gains with asynchronous crawling using `asyncio` and `httpx`.  Engineered for robustness with comprehensive error handling, `robots.txt` compliance, and concurrent Selenium management.  Built for demanding, large-scale scraping operations.

<hr>

## 🤔 Version Selector - Choose Wisely! ✅

* **👶 Start Simple:**  Use `v1` or `v2` for basic tasks and learning.
* **🌎 English Projects:** `v3` is ready for immediate use on English websites.
* **🎯 Targeted Scraping:** `v4` is your choice for efficient, relevant data extraction.
* **🚀 Performance & Scale:** `v5` is the ultimate version for large, fast, and reliable scraping. Be aware of increased complexity.

<hr>

**🔑 API Key Reminder:**  Don't forget to set your Gemini API key! (`GEMINI_API_KEY` environment variable for v5, or directly in the code for older versions).

<hr>
