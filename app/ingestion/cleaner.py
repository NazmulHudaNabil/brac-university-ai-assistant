import os
import glob
import json
import uuid
import hashlib
from pathlib import Path
from bs4 import BeautifulSoup
import re

def compute_hash(text):
    return hashlib.md5(text.encode('utf-8')).hexdigest()

def clean_html(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
            html = f.read()
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return None

    soup = BeautifulSoup(html, 'html.parser')

    # Extract title
    title = ""
    if soup.title:
        title = soup.title.string
    if not title:
        og_title = soup.find('meta', property='og:title')
        if og_title:
            title = og_title.get('content', '')
    title = title.strip() if title else os.path.basename(filepath)

    # Extract source URL
    source_url = ""
    og_url = soup.find('meta', property='og:url')
    if og_url:
        source_url = og_url.get('content', '')
    if not source_url:
        canonical = soup.find('link', rel='canonical')
        if canonical:
            source_url = canonical.get('href', '')
    if not source_url:
        # Fallback to reconstructing from filepath
        rel_path = os.path.relpath(filepath, 'data/raw/scraped')
        source_url = "https://www.bracu.ac.bd/" + rel_path.replace('.html', '')

    # Strip nav, footer, js, css
    for element in soup(['script', 'style', 'nav', 'footer', 'header', 'aside', 'noscript']):
        element.decompose()

    # Cleaned content
    # We will save the cleaned HTML to be parsed by Langchain in Phase 2
    cleaned_content = str(soup)
    
    return {
        "title": title,
        "source_url": source_url,
        "content": cleaned_content,
        "document_type": "html"
    }

def process_files():
    raw_dir = Path("data/raw/scraped")
    cleaned_dir = Path("data/cleaned")
    cleaned_dir.mkdir(parents=True, exist_ok=True)

    seen_hashes = set()
    processed_count = 0
    duplicate_count = 0

    # For now, we process HTML files. Other types like PDF can be copied over with metadata.
    for ext in ['**/*.html', '**/*.pdf', '**/*.docx', '**/*.pptx', '**/*.txt']:
        for filepath in raw_dir.glob(ext):
            if filepath.is_dir():
                continue

            file_ext = filepath.suffix.lower()
            if file_ext == '.html':
                data = clean_html(filepath)
                if not data:
                    continue
                content_hash = compute_hash(data['content'])
                # Deduplication
                if content_hash in seen_hashes:
                    duplicate_count += 1
                    continue
                seen_hashes.add(content_hash)
                
                doc_id = str(uuid.uuid4())
                metadata = {
                    "document_id": doc_id,
                    "title": data['title'],
                    "source_url": data['source_url'],
                    "document_type": "html"
                }
                
                # Save cleaned html and metadata
                out_html_path = cleaned_dir / f"{doc_id}.html"
                out_meta_path = cleaned_dir / f"{doc_id}.json"
                
                with open(out_html_path, 'w', encoding='utf-8') as f:
                    f.write(data['content'])
                with open(out_meta_path, 'w', encoding='utf-8') as f:
                    json.dump(metadata, f, indent=2)
                
                processed_count += 1
            else:
                # Handle binary/other files: just copy and create metadata
                try:
                    with open(filepath, 'rb') as f:
                        content_bytes = f.read()
                    content_hash = hashlib.md5(content_bytes).hexdigest()
                    if content_hash in seen_hashes:
                        duplicate_count += 1
                        continue
                    seen_hashes.add(content_hash)
                    
                    doc_id = str(uuid.uuid4())
                    doc_type = file_ext.replace('.', '')
                    rel_path = os.path.relpath(filepath, raw_dir)
                    source_url = "https://www.bracu.ac.bd/" + rel_path
                    
                    metadata = {
                        "document_id": doc_id,
                        "title": filepath.name,
                        "source_url": source_url,
                        "document_type": doc_type
                    }
                    
                    out_file_path = cleaned_dir / f"{doc_id}{file_ext}"
                    out_meta_path = cleaned_dir / f"{doc_id}.json"
                    
                    with open(out_file_path, 'wb') as f:
                        f.write(content_bytes)
                    with open(out_meta_path, 'w', encoding='utf-8') as f:
                        json.dump(metadata, f, indent=2)
                        
                    processed_count += 1
                except Exception as e:
                    print(f"Error processing binary {filepath}: {e}")

    print(f"Phase 1 complete. Processed: {processed_count}, Duplicates skipped: {duplicate_count}")

if __name__ == "__main__":
    process_files()
