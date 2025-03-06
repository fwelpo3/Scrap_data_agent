from crawl4ai import WebCrawler
from crawl4ai.chunking_strategy import RegexChunking

# Erstelle eine Instanz des WebCrawlers
crawler = WebCrawler()

# Optional: Konfiguriere den Crawler (z. B. User-Agent, Timeout)
crawler.warmup()

# URL, die gecrawlt werden soll
url = "https://tawil-media.de/"

try:
    # Crawle die Webseite
    result = crawler.crawl(url=url)

    # Überprüfe, ob das Crawling erfolgreich war
    if result.success:
        print("Erfolgreich gecrawlt!")
        print(f"Titel der Seite: {result.title}")
        print(f"HTML-Auszug (erster 200 Zeichen): {result.html[:200]}...")

        # Teile den Inhalt in Chunks mit der RegexChunking-Strategie
        chunker = RegexChunking(chunk_size=500)  # Chunk-Größe in Zeichen
        chunks = chunker.chunk(result.cleaned_html)

        # Gebe die ersten paar Chunks aus
        print("\nErste 3 Chunks:")
        for i, chunk in enumerate(chunks[:3]):
            print(f"Chunk {i+1}: {chunk[:100]}...")

    else:
        print(f"Crawling fehlgeschlagen: {result.error_message}")

except Exception as e:
    print(f"Ein Fehler ist aufgetreten: {str(e)}")
