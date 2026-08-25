import os
import glob
import json
import argparse
from pathlib import Path

from app.ingestion.loaders.factory import load_document
from app.ingestion.chunking.structure_aware_chunker import chunk_documents, get_chunker

def process_data(input_dir: str, output_dir: str):
    """
    Reads metadata from input_dir, loads the corresponding documents,
    chunks them, and saves the chunks into output_dir as JSON.
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # We find all .json metadata files in the cleaned directory
    metadata_files = list(input_path.glob("*.json"))
    
    if not metadata_files:
        print(f"No JSON metadata files found in {input_dir}")
        return

    chunker = get_chunker(chunk_size=1000, chunk_overlap=200)
    
    total_docs = 0
    total_chunks = 0
    
    for meta_file in metadata_files:
        documents = load_document(str(meta_file))
        if not documents:
            continue
            
        chunks = chunk_documents(documents, chunker=chunker)
        if not chunks:
            continue
            
        total_docs += 1
        total_chunks += len(chunks)
        
        # We can extract the doc_id from the metadata of the first chunk
        doc_id = chunks[0].metadata.get("document_id")
        
        # Convert chunks to a list of dicts to save as JSON
        chunks_data = []
        for i, chunk in enumerate(chunks):
            # inject a chunk id
            chunk_metadata = chunk.metadata.copy()
            chunk_metadata["chunk_id"] = f"{doc_id}_{i}"
            
            chunks_data.append({
                "page_content": chunk.page_content,
                "metadata": chunk_metadata
            })
            
        out_file = output_path / f"{doc_id}_chunks.json"
        with open(out_file, 'w', encoding='utf-8') as f:
            json.dump(chunks_data, f, indent=2)

    print(f"Ingestion Phase 2 complete. Processed {total_docs} documents into {total_chunks} chunks.")
    print(f"Saved chunks to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the Phase 2 Ingestion Pipeline.")
    parser.add_argument("input_dir", type=str, nargs='?', default="data/cleaned",
                        help="Path to the cleaned data directory")
    parser.add_argument("--output", type=str, default="processed_data",
                        help="Path to save the chunked JSON files")
    
    args = parser.parse_args()
    process_data(args.input_dir, args.output)
