"""
PDF downloader tool for downloading PDF files from URLs.

This module provides a standalone PDFDownloader class that downloads PDF files
from URLs and saves them to disk. It's designed as an independent tool that can
be used in any context, not just the RAG workflow.

The downloader is decoupled from the workflow architecture, following the
principle of tool independence. It can be used standalone or integrated into
workflow nodes via explicit dependency injection.
"""
from typing import Optional
import os
import urllib.request
import urllib.error
import ssl
from urllib.parse import urlparse
from pathlib import Path


class PDFDownloader:
    """
    Standalone tool for downloading PDF files from URLs.
    
    This class provides a simple interface for downloading PDF files from URLs
    and saving them to a local directory. It's designed to be:
    - Independent: No dependencies on workflow, RAG, or vector DB
    - Reusable: Can be used in any context
    - Robust: Handles errors gracefully
    
    Example:
        .. code-block:: python
        
            downloader = PDFDownloader()
            pdf_path = downloader.download(
                "http://arxiv.org/pdf/1234.5678.pdf",
                download_dir="./papers"
            )
            if pdf_path:
                print(f"Downloaded to {pdf_path}")
    """
    
    def download(
        self,
        pdf_url: str,
        download_dir: str = "./papers"
    ) -> Optional[str]:
        """
        Download a PDF file from a URL and save it to disk.
        
        Downloads the PDF file from the given URL and saves it to the specified
        directory. The filename is derived from the URL (typically the last part
        of the path or arxiv ID).
        
        Args:
            pdf_url: URL to the PDF file. Must be a valid HTTP/HTTPS URL.
            download_dir: Directory path where the PDF should be saved.
                Directory is created if it doesn't exist. Default is "./papers".
        
        Returns:
            Path to the downloaded PDF file as a string, or None if download fails.
            The file path is relative to the current working directory or absolute
            if an absolute path is provided.
        
        Example:
            .. code-block:: python
            
                downloader = PDFDownloader()
                
                # Download from arxiv
                pdf_path = downloader.download(
                    "http://arxiv.org/pdf/1234.5678.pdf",
                    download_dir="./downloads"
                )
                
                if pdf_path:
                    print(f"Downloaded to: {pdf_path}")
                else:
                    print("Download failed")
        
        Raises:
            No exceptions are raised. All errors are handled internally and
            logged. Returns None on failure.
        
        Note:
            - Directory is created automatically if it doesn't exist
            - Filename is extracted from URL path or arxiv ID
            - Network errors, invalid URLs, and file system errors are caught
            - Returns None silently on any error (error is printed to console)
            - Does not validate that downloaded content is actually a PDF
        """
        try:
            # Create download directory if it doesn't exist
            os.makedirs(download_dir, exist_ok=True)
            
            # Parse URL to extract filename
            parsed_url = urlparse(pdf_url)
            
            # Extract filename from URL path
            # For arxiv URLs: /pdf/1234.5678.pdf -> 1234.5678.pdf
            url_path = parsed_url.path
            if url_path:
                # Get the last component of the path
                filename = os.path.basename(url_path)
                
                # If no extension, assume .pdf
                if not filename or not filename.endswith('.pdf'):
                    # Try to extract arxiv ID from path
                    if '/pdf/' in url_path:
                        arxiv_id = url_path.split('/pdf/')[-1].split('/')[0]
                        filename = f"{arxiv_id}.pdf"
                    else:
                        # Fallback: use last part of path or generate name
                        filename = url_path.split('/')[-1] or "downloaded.pdf"
                        if not filename.endswith('.pdf'):
                            filename += '.pdf'
            else:
                # No path in URL, generate filename from domain or use default
                filename = "downloaded.pdf"
            
            # Construct full file path
            file_path = os.path.join(download_dir, filename)
            
            # Download the file
            print(f"Downloading PDF from {pdf_url}...")
            
            # Create SSL context that doesn't verify certificates (for compatibility)
            # In production, you might want proper certificate verification
            ssl_context = ssl.create_default_context()
            ssl_context.check_hostname = False
            ssl_context.verify_mode = ssl.CERT_NONE
            
            # Use urlretrieve with SSL context
            opener = urllib.request.build_opener(urllib.request.HTTPSHandler(context=ssl_context))
            urllib.request.install_opener(opener)
            urllib.request.urlretrieve(pdf_url, file_path)
            
            # Verify file was created
            if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
                print(f"✓ Downloaded PDF to {file_path}")
                return file_path
            else:
                print(f"✗ Download failed: File not created or empty")
                return None
                
        except urllib.error.URLError as e:
            print(f"✗ Download failed: Network error - {e}")
            return None
        except urllib.error.HTTPError as e:
            print(f"✗ Download failed: HTTP error {e.code} - {e.reason}")
            return None
        except ValueError as e:
            print(f"✗ Download failed: Invalid URL - {e}")
            return None
        except OSError as e:
            print(f"✗ Download failed: File system error - {e}")
            return None
        except Exception as e:
            print(f"✗ Download failed: Unexpected error - {e}")
            return None

