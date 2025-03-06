import os
import json
import logging
from typing import List, Optional

import google.generativeai as genai


# Konfiguration für Gemini
class GeminiConfig:
    api_key: str = "AIzaSyC9N8ydc_4IdCybpJy5i3xgohCkKIJd6QE"  # ERSETZEN MIT DEINEM ECHTEN API-KEY
    model_name: str = "gemini-2.0-pro-exp-02-05" # or "gemini-2.0-flash-thinking-exp-01-21" or other model


def prioritize_links_with_gemini(links: List[str], priority_instructions: str = None) -> Optional[List[str]]:
    """
    Priorisiert eine Liste von Links mithilfe des Gemini-Modells und behandelt Markdown-formatierte JSON-Antworten.
    """
    config = GeminiConfig() # Nutzt die GeminiConfig Klasse
    genai.configure(api_key=config.api_key)
    model = genai.GenerativeModel(model_name=config.model_name)

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
        response = model.generate_content(prompt_content)
        response.resolve() # wait for response to be available

        if response.parts:
            response_text = response.text
            json_str = response_text  # Initialisiere json_str mit dem gesamten Antworttext

            # Versuche, JSON aus Markdown-Codeblöcken zu extrahieren
            if "```json" in response_text:
                try:
                    json_str = response_text.split("```json")[1].split("```")[0].strip()
                except IndexError:
                    logging.warning("Ungültige Markdown-JSON-Antwortstruktur erkannt, versuche es mit einfacher ``` Extraktion.")
                    try: # Fallback für einfache ``` Codeblöcke (ohne json Kennzeichnung)
                        json_str = response_text.split("```")[1].split("```")[0].strip()
                    except IndexError:
                        logging.error("Kein JSON-Codeblock in der Gemini-Antwort gefunden.")
                        logging.error(f"Gemini Antworttext: {response_text}")
                        return None
            elif "```" in response_text: # Fallback für einfache ``` Codeblöcke (ohne json Kennzeichnung)
                 try:
                    json_str = response_text.split("```")[1].split("```")[0].strip()
                 except IndexError:
                    logging.error("Ungültige einfache Markdown-Codeblock Struktur erkannt.")
                    logging.error(f"Gemini Antworttext: {response_text}")
                    return None


            try:
                sorted_links_json = json.loads(json_str)
                if isinstance(sorted_links_json, list): # Stelle sicher, dass es eine Liste ist
                    logging.info("Links erfolgreich von Gemini priorisiert.")
                    return sorted_links_json
                else:
                    logging.error(f"Unerwartetes JSON-Format von Gemini erhalten. Erwartete Liste, bekam: {type(sorted_links_json)}")
                    logging.error(f"Gemini Antworttext (extrahiert): {json_str}") # Logge den extrahierten JSON-String
                    return None
            except json.JSONDecodeError as e:
                logging.error(f"Fehler beim Parsen der JSON-Antwort von Gemini: {e}")
                logging.error(f"Gemini Antworttext (extrahiert): {json_str}") # Logge den extrahierten JSON-String
                return None
        else:
            logging.error("Leere Antwort von Gemini erhalten.")
            return None


    except Exception as e:
        logging.error(f"Fehler bei der Kommunikation mit Gemini oder Verarbeitung der Antwort: {e}")
        return None


if __name__ == '__main__':
    # Beispielhafte Nutzung der Funktion
    example_links = [
        "https://www.example.com/ueber-uns", # Über uns Seite
        "https://www.example.com/produkte", # Produktseite
        "https://www.example.com/kontakt",  # Kontaktseite
        "https://www.example.com/team/john-doe", # Profilseite einer Person
        "https://www.example.com/blog" # Blog Seite
    ]

    # Beispiel 1: Standardmäßige Priorisierung für persönliche Informationen
    print("Standardmäßige Priorisierung für persönliche Informationen:")
    sorted_links_default = prioritize_links_with_gemini(example_links)
    if sorted_links_default:
        print(json.dumps(sorted_links_default, indent=2, ensure_ascii=False))
    else:
        print("Fehler bei der Priorisierung mit Gemini.")

    print("\n---")

    # Beispiel 2: Mit spezifischen Anweisungen
    custom_instructions = "Priorisiere die folgenden Links basierend auf ihrer Relevanz für das Auffinden von Kontaktinformationen wie E-Mail-Adressen und Telefonnummern. Sortiere sie so, dass die wahrscheinlichsten Links für Kontaktinformationen zuerst stehen. Antworte mit einer JSON-Liste der sortierten Links."
    print("Priorisierung für Kontaktinformationen mit spezifischen Anweisungen:")
    sorted_links_custom = prioritize_links_with_gemini(example_links, custom_instructions)
    if sorted_links_custom:
        print(json.dumps(sorted_links_custom, indent=2, ensure_ascii=False))
    else:
        print("Fehler bei der Priorisierung mit Gemini.")