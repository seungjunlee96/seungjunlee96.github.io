#!/usr/bin/env python3
"""
Citation Updater using Semantic Scholar API
Automatically updates citation counts for publications in index.md
"""

import requests
import re
import time
from typing import Optional

class CitationUpdater:
    """Updates citation counts using Semantic Scholar API."""
    
    BASE_URL = "https://api.semanticscholar.org/graph/v1/paper"
    HEADERS = {
        "User-Agent": "CitationUpdater/1.0 (https://seungjunlee96.github.io)"
    }
    
    def __init__(self, cv_path: str = "index.md"):
        self.cv_path = cv_path
        
    def get_citation_count_by_doi(self, doi: str) -> Optional[int]:
        """Get citation count using DOI."""
        try:
            url = f"{self.BASE_URL}/DOI:{doi}"
            params = {"fields": "citationCount"}
            
            response = requests.get(url, headers=self.HEADERS, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            return data.get("citationCount")
        except Exception as e:
            print(f"⚠ Error fetching DOI {doi}: {e}")
            return None
    
    def get_citation_count_by_title(self, title: str) -> Optional[int]:
        """Get citation count by searching with title."""
        try:
            # Search API
            search_url = "https://api.semanticscholar.org/graph/v1/paper/search"
            params = {
                "query": title[:200],  # Limit query length
                "fields": "citationCount,title",
                "limit": 5
            }
            
            response = requests.get(search_url, headers=self.HEADERS, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            papers = data.get("data", [])
            
            # Find best match by title similarity
            title_lower = title.lower()
            for paper in papers:
                paper_title = paper.get("title", "").lower()
                # Simple similarity check - check if significant portion matches
                if title_lower in paper_title or paper_title in title_lower:
                    citation_count = paper.get("citationCount")
                    if citation_count is not None:
                        return citation_count
            
            # If no exact match, return first result if available
            if papers:
                citation_count = papers[0].get("citationCount")
                if citation_count is not None:
                    return citation_count
                
            return None
        except Exception as e:
            print(f"⚠ Error searching for '{title[:50]}...': {e}")
            return None
    
    def get_citation_count(self, doi: Optional[str], title: str) -> Optional[int]:
        """Get citation count using DOI first, then title search."""
        # Try DOI first if available
        if doi:
            citation_count = self.get_citation_count_by_doi(doi)
            if citation_count is not None:
                return citation_count
        
        # Fallback to title search
        return self.get_citation_count_by_title(title)
    
    def update_citations(self) -> bool:
        """Update citation counts in index.md."""
        # Paper metadata with DOI and unique URL identifier
        papers = [
            {
                "title": "Automated Idiopathic Normal Pressure Hydrocephalus Diagnosis via Artificial Intelligence–Based 3D T1 MRI Volumetric Analysis",
                "doi": "10.3174/ajnr.A8489",
                "url_identifier": "ajnr.A8489",  # Unique part of URL
            },
            {
                "title": "Emergency Triage of Brain Computed Tomography via Anomaly Detection with a Deep Generative Model",
                "doi": "10.1038/s41467-022-31808-0",
                "url_identifier": "s41467-022-31808-0",  # Unique part of URL
            },
            {
                "title": "Enhancement of Evaluating Flatfoot on a Weight-Bearing Lateral Radiograph of the Foot with U-Net Based Semantic Segmentation on the Long Axis of Tarsal and Metatarsal Bones",
                "doi": "10.1016/j.compbiomed.2022.105400",
                "url_identifier": "S0010482522001925",  # Unique part of URL
            },
            {
                "title": "Enhancing Deep Learning Based Classifiers with Inpainting Anatomical Side Markers (L/R Markers) for Multi-Center Trials",
                "doi": "10.1016/j.cmpb.2022.106705",
                "url_identifier": "S0169260722000906",  # Unique part of URL
            },
            {
                "title": "Deep Learning on Radar-Centric 3D Object Detection",
                "doi": None,  # arXiv paper, no DOI
                "url_identifier": "2003.00851",  # arXiv ID
            },
        ]
        
        try:
            with open(self.cv_path, 'r', encoding='utf-8') as file:
                content = file.read()
        except FileNotFoundError:
            print(f"❌ Error: File '{self.cv_path}' not found")
            return False
        except Exception as e:
            print(f"❌ Error reading file: {e}")
            return False
        
        changes_made = False
        
        for paper in papers:
            title = paper["title"]
            doi = paper.get("doi")
            url_id = paper.get("url_identifier")
            
            # Get citation count first
            citation_count = self.get_citation_count(doi, title)
            
            if citation_count is None:
                print(f"⚠ Could not find citation count for: {title[:60]}...")
                time.sleep(1)
                continue
            
            # Use URL identifier to find the exact publication item
            # This is more reliable than title matching
            url_pattern = re.escape(url_id)
            
            # Pattern to find the publication item div containing this URL
            # Match from opening div to closing div, ensuring we get the complete item
            pattern = rf'(<div class="publication-item[^"]*">.*?{url_pattern}.*?)(</div>)'
            
            match = re.search(pattern, content, re.DOTALL | re.IGNORECASE)
            
            if match:
                paper_section = match.group(1)
                closing_div = match.group(2)
                
                # Check if citation already exists
                citation_pattern = r'<span class="citation-count">📊 \d+ citations</span>'
                has_citation = re.search(citation_pattern, paper_section)
                
                citation_html = f' | <span class="citation-count">📊 {citation_count} citations</span>'
                
                # Create replacement pattern that includes the URL identifier to ensure uniqueness
                # This ensures we only replace the citation for this specific paper
                if has_citation:
                    # Update existing citation - use URL identifier in pattern to ensure uniqueness
                    unique_pattern = rf'({re.escape(url_id)}.*?<span class="citation-count">📊 )\d+( citations</span>)'
                    replacement = rf'\g<1>{citation_count}\g<2>'
                    content = re.sub(unique_pattern, replacement, content, flags=re.DOTALL | re.IGNORECASE)
                else:
                    # Add citation count before closing div - use URL identifier to find exact location
                    unique_pattern = rf'({re.escape(url_id)}.*?</a>)(\s*</div>)'
                    replacement = rf'\g<1>{citation_html}\g<2>'
                    content = re.sub(unique_pattern, replacement, content, flags=re.DOTALL | re.IGNORECASE)
                
                print(f"✓ Updated: {title[:60]}... - {citation_count} citations")
                changes_made = True
            else:
                print(f"⚠ Could not find publication item for: {title[:60]}... (URL: {url_id})")
            
            # Rate limiting: be nice to the API
            time.sleep(1)
        
        if changes_made:
            try:
                with open(self.cv_path, 'w', encoding='utf-8') as file:
                    file.write(content)
                print("✓ File updated successfully")
                return True
            except Exception as e:
                print(f"❌ Error writing file: {e}")
                return False
        else:
            print("ℹ No changes to citations")
            return False

if __name__ == "__main__":
    updater = CitationUpdater()
    updater.update_citations()
