import requests
from bs4 import BeautifulSoup
import re

def get_citation_count(scholar_profile_url, paper_title):
    response = requests.get(scholar_profile_url)
    soup = BeautifulSoup(response.content, 'html.parser')

    # Find the paper element by title
    paper_element = soup.find('a', text=paper_title)

    # If paper is not found, return None
    if not paper_element:
        return None

    # Get the citation count for the paper
    citation_element = paper_element.find_next('a', text=re.compile(r'Cited by \d+'))

    # If citation element is not found, return 0
    if not citation_element:
        return 0

    citation_count = int(re.search(r'\d+', citation_element.text).group())
    return citation_count

scholar_profile_url = 'https://scholar.google.com/citations?user=VfYHEWgAAAAJ&hl=en&authuser=1'

paper_titles = [
    'Deep learning on radar centric 3D object detection',
    'Emergency Triage of Brain Computed Tomography via Anomaly Detection with a Deep Generative Model',
]
citation_counts = {
    paper_title: get_citation_count(scholar_profile_url, paper_title)
    for paper_title in paper_titles
}

# Now, open your CV markdown and replace the placeholder with the citation count
for paper_title, citation_count in citation_counts.items():
    with open('./_posts/about/2023-10-04-about.md', 'r') as file:
        cv_content = file.read()

    updated_content = re.sub(rf'CITED BY: \[{paper_title}\]', f'CITED BY: {citation_count}', cv_content)

    with open('./_posts/about/2023-10-04-about.md', 'w') as file:
        file.write(updated_content)
