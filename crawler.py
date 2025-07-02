import os
import time
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse

# Configuration
base_url = "https://uiet.puchd.ac.in/wp-content/uploads/"
download_dir = "uiet_pdfs"
crawl_delay = 5  # Increased delay to 5 seconds
timeout = 30  # Increased timeout to 30 seconds
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8"
}

def is_valid_url(url):
    parsed = urlparse(url)
    return parsed.netloc == "uiet.puchd.ac.in"

def download_pdf(url):
    try:
        response = requests.get(
            url,
            headers=headers,
            timeout=timeout,
            stream=True
        )
        response.raise_for_status()
        
        # Extract filename
        filename = os.path.basename(urlparse(url).path) or "file.pdf"
        if not filename.lower().endswith('.pdf'):
            filename += '.pdf'
        
        # Save file
        filepath = os.path.join(download_dir, filename)
        with open(filepath, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:  # filter out keep-alive chunks
                    f.write(chunk)
        print(f"✅ Downloaded: {filename}")
        return True
        
    except Exception as e:
        print(f"❌ Failed to download {url}: {str(e)}")
        return False

def test_connection():
    """Test connection to the website"""
    print("Testing connection to server...")
    try:
        response = requests.head(
            base_url, 
            headers=headers, 
            timeout=timeout,
            allow_redirects=True
        )
        print(f"Connection test: HTTP {response.status_code}")
        return response.status_code == 200
    except Exception as e:
        print(f"Connection failed: {str(e)}")
        return False

def crawl_site(start_url):
    visited = set()
    queue = [start_url]
    pdf_count = 0
    
    while queue:
        url = queue.pop(0)
        
        if url in visited:
            continue
        visited.add(url)
        
        try:
            print(f"🔍 Crawling: {url}")
            response = requests.get(
                url, 
                headers=headers, 
                timeout=timeout
            )
            response.raise_for_status()
            
            # Check if content is HTML before parsing
            content_type = response.headers.get('Content-Type', '')
            if 'text/html' not in content_type:
                print(f"⚠️ Skipping non-HTML content: {content_type}")
                continue
                
            soup = BeautifulSoup(response.text, 'html.parser')
            
            for link in soup.find_all('a', href=True):
                href = link['href'].strip()
                absolute_url = urljoin(url, href)
                
                if not is_valid_url(absolute_url):
                    continue
                
                # Handle PDF links
                if absolute_url.lower().endswith('.pdf'):
                    if absolute_url not in visited:
                        print(f"📄 Found PDF: {absolute_url}")
                        time.sleep(crawl_delay)
                        if download_pdf(absolute_url):
                            pdf_count += 1
                            visited.add(absolute_url)
                    continue
                
                # Queue new pages
                if absolute_url not in visited and absolute_url not in queue:
                    queue.append(absolute_url)
            
            time.sleep(crawl_delay)
            
        except requests.exceptions.Timeout:
            print(f"⌛ Timeout occurred for {url}")
        except Exception as e:
            print(f"⚠️ Error processing {url}: {str(e)}")
    
    return pdf_count

def main():
    os.makedirs(download_dir, exist_ok=True)
    print(f"Created download directory: {download_dir}")
    
    # Test connection before crawling
    if not test_connection():
        print("\n❌ Connection test failed! Possible reasons:")
        print("- Network firewall blocking requests")
        print("- Website is down or unreachable")
        print("- Server blocking automated requests")
        print("\nTry these solutions:")
        print("1. Check your internet connection")
        print("2. Visit the site manually in browser")
        print("3. Use VPN if network blocks requests")
        print("4. Increase timeout value in the script")
        print("5. Contact site administrator if server is down")
        return
    
    print(f"\nStarting crawl of {base_url}...")
    total_pdfs = crawl_site(base_url)
    print(f"\nCrawl complete! Downloaded {total_pdfs} PDF files to '{download_dir}'")

if __name__ == "__main__":
    main()