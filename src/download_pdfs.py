import os
import json
import time
import requests
from loguru import logger

def download_pdfs() -> None:
    """
    Downloads PDFs for papers defined in papers.json and maintains a download manifest.
    Respects arXiv rate limits by inserting delays between downloads.
    """
    pdf_dir = "data/raw/pdfs"
    os.makedirs(pdf_dir, exist_ok=True)
    
    papers_json_path = "data/raw/papers.json"
    manifest_path = os.path.join(pdf_dir, "download_manifest.json")
    
    if not os.path.exists(papers_json_path):
        logger.error(f"Source papers list metadata not found at {papers_json_path}")
        return
        
    with open(papers_json_path, "r", encoding="utf-8") as f:
        papers_data = json.load(f)
        
    papers_list = papers_data.get("papers", [])
    logger.info(f"Loaded {len(papers_list)} papers from papers.json.")
    
    # Load existing download manifest if present
    manifest = {}
    if os.path.exists(manifest_path):
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest = json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load download manifest: {e}. Starting fresh.")
            
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    
    downloaded_count = 0
    skipped_count = 0
    failed_count = 0
    
    for paper in papers_list:
        arxiv_id = paper.get("arxiv_id")
        pdf_url = paper.get("pdf_url")
        
        if not arxiv_id or not pdf_url:
            continue
            
        local_filename = f"{arxiv_id}.pdf"
        local_path = os.path.join(pdf_dir, local_filename)
        
        # Check if local file exists and manifest lists it as success
        if os.path.exists(local_path) and arxiv_id in manifest and manifest[arxiv_id].get("status") == "success":
            logger.debug(f"Skipping {arxiv_id}, already downloaded.")
            skipped_count += 1
            continue
            
        logger.info(f"Downloading {arxiv_id} from {pdf_url}...")
        
        success = False
        retries = 3
        for attempt in range(1, retries + 1):
            try:
                # Add explicit delay to prevent IP blocking by arXiv
                time.sleep(1.5)
                response = requests.get(pdf_url, headers=headers, timeout=30)
                if response.status_code == 200:
                    with open(local_path, "wb") as pdf_file:
                        pdf_file.write(response.content)
                    success = True
                    break
                elif response.status_code == 403:
                    logger.warning(f"Forbidden (403) on attempt {attempt} for {arxiv_id}. Applying backoff...")
                    time.sleep(5 * attempt)
                else:
                    logger.warning(f"Attempt {attempt} returned status code {response.status_code} for {arxiv_id}")
            except Exception as e:
                logger.warning(f"Attempt {attempt} raised exception for {arxiv_id}: {e}")
                time.sleep(2 * attempt)
                
        if success:
            logger.info(f"Successfully downloaded {arxiv_id}.")
            manifest[arxiv_id] = {
                "status": "success",
                "local_path": local_path,
                "downloaded_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "title": paper.get("title")
            }
            downloaded_count += 1
        else:
            logger.error(f"Failed to download {arxiv_id} after {retries} attempts.")
            manifest[arxiv_id] = {
                "status": "failed",
                "error": "Failed after all retries",
                "downloaded_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "title": paper.get("title")
            }
            failed_count += 1
            
        # Update manifest file on disk after each attempt
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
            
    logger.info("=" * 60)
    logger.info("PDF DOWNLOAD COMPLETED SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Downloaded: {downloaded_count}")
    logger.info(f"Skipped   : {skipped_count}")
    logger.info(f"Failed    : {failed_count}")
    logger.info(f"Total     : {len(papers_list)}")
    logger.info("=" * 60)

if __name__ == "__main__":
    download_pdfs()
