V1:
Trying to make an konzept to scrape the Information's from the Web, it has sever Problems, Like it, scrape one page and sends it to the llm etc.

V2:
- First working Version, it searches on a given Website and returns Json File.
- It scrape the Content and sends it at the end to the LLM
- The Prompt can be modified so, it searches just for one Person
Problem: it may return a json file with a lot of infos and people based on the Website

V3:
- Promblem solved, it returns an array of Persons if there ist more then one

V4_beta:
Idea:
1. Get all links from a url,
2. Send Links to llm to prioritize them based on the information that we want and returns a some of them
3. Scrape the returned Links
4. Send to llm for json extraction
