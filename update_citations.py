#!/usr/bin/env python3
"""
Citation Updater for Markdown CV
Updates citation counts using Semantic Scholar API for publications in index.md
"""

import requests
import re
import time
from typing import Optional

class CitationUpdater:
    """Updates citation counts using Semantic Scholar API."""
    
    BASE_URL = "https://api.semanticscholar.org/graph/v1/paper"
    HEADERS = {
        "User-Agent": "CitationUpdater/2.0 (mailto:lsjj096@gmail.com)"
    }
    
    def __init__(self, cv_path: str = "index.md"):
        self.cv_path = cv_path
        
    def get_citation_count(self, identifier: str, id_type: str = 'DOI') -> Optional[int]:
        """
        Get citation count using DOI or arXiv ID.
        identifier: The DOI (e.g. '10.1038/s41467...') or arXiv ID (e.g. '2003.00851')
        id_type: 'DOI' or 'ARXIV'
        """
        try:
            # Semantic Scholar supports DOI:prefix or ARXIV:prefix
            query_id = f"{id_type}:{identifier}"
            url = f"{self.BASE_URL}/{query_id}"
            params = {"fields": "citationCount,title"}
            
            response = requests.get(url, headers=self.HEADERS, params=params, timeout=10)
            
            # Handle 404 cleanly
            if response.status_code == 404:
                print(f"⚠ Paper not found in API: {identifier}")
            return None
    
            response.raise_for_status()
            data = response.json()
            
            count = data.get("citationCount")
            title = data.get("title", "Unknown Title")
            print(f"   Found: {count} citations for '{title[:50]}...'")
            return count
            
        except Exception as e:
            print(f"⚠ Error fetching {identifier}: {e}")
            return None
    
    def update_citations(self) -> bool:
        """Update citation counts in the markdown file."""
        
        # Define your papers here.
        # 'unique_url_part': A string unique to the URL in your index.md to identify the line.
        # 'id': The DOI or ArXiv ID for the API.
        papers = [
            {
                "id": "10.3174/ajnr.A8489",
                "type": "DOI",
                "unique_url_part": "ajnr.A8489",
            },
            {
                "id": "10.1038/s41467-022-31808-0",
                "type": "DOI",
                "unique_url_part": "s41467-022-31808-0",
            },
            {
                "id": "10.1016/j.compbiomed.2022.105400",
                "type": "DOI",
                "unique_url_part": "S0010482522001925",
            },
            {
                "id": "10.1016/j.cmpb.2022.106705",
                "type": "DOI",
                "unique_url_part": "S0169260722000906",  # Matches the ScienceDirect URL ID
            },
            {
                "id": "2003.00851",
                "type": "ARXIV",
                "unique_url_part": "2003.00851",
            },
        ]
        
        try:
            with open(self.cv_path, 'r', encoding='utf-8') as file:
                lines = file.readlines()
        except FileNotFoundError:
            print(f"❌ Error: File '{self.cv_path}' not found")
            return False
        
        changes_made = False
        new_lines = []

        for line in lines:
            updated_line = line
            
            # Check if this line contains one of our papers
            for paper in papers:
                if paper["unique_url_part"] in line:
                    print(f"🔍 Processing paper: {paper['unique_url_part']}...")
                    
                    count = self.get_citation_count(paper["id"], paper["type"])
                    
                    if count is not None:
                        # Remove existing citation count if present (handles with or without emoji)
                        # Pattern matches: | <span class="citation-count">(📊 )?X citations</span>
                        clean_line = re.sub(
                            r'\s*\|\s*<span class="citation-count">(📊 )?\d+ citations</span>\s*',
                            '',
                            line.rstrip()
                        )
                        
                        # Append new count (without emoji)
                        updated_line = f"{clean_line} | <span class=\"citation-count\">{count} citations</span>\n"
                        
                        if updated_line != line:
                            changes_made = True
                            print(f"   ✓ Updated line.")
            
                    # Pause briefly to be nice to the API
                    time.sleep(0.5)
                    break  # Stop checking other papers for this line
            
            new_lines.append(updated_line)
        
        if changes_made:
            try:
                with open(self.cv_path, 'w', encoding='utf-8') as file:
                    file.writelines(new_lines)
                print("✨ File updated successfully")
                return True
            except Exception as e:
                print(f"❌ Error writing file: {e}")
                return False
        else:
            print("ℹ No changes needed.")
            return False

if __name__ == "__main__":
    updater = CitationUpdater()
    updater.update_citations()
