import requests
from bs4 import BeautifulSoup
import re
from typing import List, Dict

class CitationException(Exception):
    """Base exception for all citation related errors."""
    pass

class PaperNotFoundException(CitationException):
    """Raised when a paper is not found on Google Scholar."""
    pass

class CitationCountNotFoundException(CitationException):
    """Raised when the citation count for a paper is not found."""
    pass

class CitationUpdater:
    HEADERS = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.107 Safari/537.36"
    }

    def __init__(self, scholar_profile_url: str, cv_path: str):
        self.scholar_profile_url = scholar_profile_url
        self.cv_path = cv_path

    def get_citation_count(self, soup: BeautifulSoup, paper_title: str) -> int:
        paper_element = soup.find('a', string=paper_title)
        if not paper_element:
            raise PaperNotFoundException(f"'{paper_title}' not found.")
        
        citation_element = paper_element.find_next(string=re.compile(r'^\d+$'))
        if not citation_element:
            raise CitationCountNotFoundException("Citation count not found.")
        
        return int(citation_element.strip())

    def update_cv_with_citation_counts(self, paper_titles: List[str]) -> None:
        try:
            soup = self._get_soup(self.scholar_profile_url)
        except Exception as e:
            raise CitationException(f"Failed to fetch Google Scholar page: {e}")

        try:
            with open(self.cv_path, 'r') as file:
                cv_content = file.read()
        except FileNotFoundError:
            raise CitationException(f"CV file not found: {self.cv_path}")
        except Exception as e:
            raise CitationException(f"Failed to read CV file: {e}")

        for paper_title in paper_titles:
            try:
                citation_count = self.get_citation_count(soup, paper_title)
                old_text = f'Cited by **[<citation>]**'.replace('<citation>', paper_title)
                new_text = f'Cited by **{citation_count}**'
                if old_text in cv_content:
                    cv_content = cv_content.replace(old_text, new_text)
                    print(f"Updated: {paper_title} - Citations: {citation_count}")
                else:
                    print(f"Warning: Could not find citation placeholder for '{paper_title}'")
            except PaperNotFoundException as e:
                print(f"Warning: {e}")
            except CitationCountNotFoundException as e:
                print(f"Warning: {e}")

        try:
            with open(self.cv_path, 'w') as file:
                file.write(cv_content)
        except Exception as e:
            raise CitationException(f"Failed to write CV file: {e}")

    def _get_soup(self, url: str) -> BeautifulSoup:
        try:
            response = requests.get(url, headers=self.HEADERS, timeout=10)
            response.raise_for_status()
            return BeautifulSoup(response.content, 'html.parser')
        except requests.exceptions.RequestException as e:
            raise CitationException(f"Failed to fetch page: {e}")

if __name__ == "__main__":
    CV_PATH = '_posts/about/2023-10-04-about.md'
    SCHOLAR_PROFILE_URL = 'https://scholar.google.com/citations?user=VfYHEWgAAAAJ&hl=en&authuser=1'
    paper_titles = [
        'Deep learning on radar centric 3D object detection',
        'Emergency triage of brain computed tomography via anomaly detection with a deep generative model',
    ]

    updater = CitationUpdater(SCHOLAR_PROFILE_URL, CV_PATH)
    updater.update_cv_with_citation_counts(paper_titles)
