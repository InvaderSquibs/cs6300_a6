"""
Quick script to check where PDFs are saved and list downloaded files.

This helps you find PDFs that have been downloaded by the PDF downloader.
"""
import os
from pathlib import Path

def check_pdf_location():
    """Show where PDFs are saved and list any existing PDFs."""
    print("="*60)
    print("PDF Download Location Check")
    print("="*60)
    print()
    
    # Default locations
    default_dir = "./papers"
    abs_default = os.path.abspath(default_dir)
    
    print(f"Default PDF directory: {default_dir}")
    print(f"Full path: {abs_default}")
    print()
    
    # Check if directory exists
    if os.path.exists(default_dir):
        print(f"✓ Directory exists: {abs_default}")
        
        # List PDFs
        pdfs = list(Path(default_dir).glob("*.pdf"))
        if pdfs:
            print(f"\nFound {len(pdfs)} PDF file(s):")
            for pdf in sorted(pdfs):
                size = pdf.stat().st_size
                size_mb = size / (1024 * 1024)
                print(f"  • {pdf.name}")
                print(f"    Path: {pdf}")
                print(f"    Size: {size:,} bytes ({size_mb:.2f} MB)")
        else:
            print("\n  No PDF files found in directory")
    else:
        print(f"⚠ Directory doesn't exist yet (will be created on first download)")
        print(f"  Will be created at: {abs_default}")
    
    # Check example_papers (from example script)
    example_dir = "./example_papers"
    if os.path.exists(example_dir):
        pdfs = list(Path(example_dir).glob("*.pdf"))
        if pdfs:
            print(f"\nExample directory ({example_dir}) contains:")
            for pdf in sorted(pdfs):
                size = pdf.stat().st_size
                print(f"  • {pdf.name} ({size:,} bytes)")
    
    print()
    print("="*60)
    print("Note: PDFs are saved relative to where you run the script")
    print("="*60)

if __name__ == "__main__":
    check_pdf_location()

