import PyPDF2
import fitz  # pymupdf
import re
from pathlib import Path
from typing import List, Dict, Tuple
import json

class PDFProcessor:
    def __init__(self, pdf_directory: str, output_dir: str = "processed_data"):
        self.pdf_dir = Path(pdf_directory)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
    def extract_text_with_formatting(self, pdf_path: Path) -> str:
        """Extract text while preserving code formatting"""
        doc = fitz.open(pdf_path)
        full_text = ""
        
        for page_num in range(doc.page_count):
            page = doc.load_page(page_num)
            text = page.get_text()
            
            # Clean up common PDF artifacts
            text = self.clean_pdf_artifacts(text)
            full_text += text + "\n\n"
            
        doc.close()
        return full_text
    
    def clean_pdf_artifacts(self, text: str) -> str:
        """Remove common PDF extraction artifacts"""
        # Remove excessive whitespace
        text = re.sub(r'\n\s*\n\s*\n', '\n\n', text)
        
        # Fix broken words (hyphenated line breaks)
        text = re.sub(r'(\w+)-\n(\w+)', r'\1\2', text)
        
        # Remove page headers/footers (basic pattern)
        text = re.sub(r'^.*?Page \d+.*?$', '', text, flags=re.MULTILINE)
        
        return text.strip()
    
    def create_instruction_pairs(self, text: str, pdf_name: str) -> List[Dict]:
        """Convert extracted text into instruction-response pairs"""
        pairs = []
        
        # Split into sections (customize based on your PDF structure)
        sections = self.split_into_sections(text)
        
        for i, section in enumerate(sections):
            if len(section.strip()) > 100:  # Skip very short sections
                # Create Q&A pairs from sections
                instruction = f"Explain the following programming concept from {pdf_name}:"
                response = section.strip()
                
                pairs.append({
                    "instruction": instruction,
                    "input": "",
                    "output": response,
                    "source": pdf_name,
                    "section": i
                })
        
        return pairs
    
    def split_into_sections(self, text: str) -> List[str]:
        """Split text into logical sections"""
        # Basic splitting by headers (customize for your content)
        sections = re.split(r'\n(?=[A-Z][^a-z\n]*\n)', text)
        
        # Filter out very short sections
        sections = [s for s in sections if len(s.strip()) > 50]
        
        return sections
    
    def process_all_pdfs(self) -> List[Dict]:
        """Process all PDFs in directory"""
        all_pairs = []
        
        for pdf_file in self.pdf_dir.glob("*.pdf"):
            print(f"Processing: {pdf_file.name}")
            
            try:
                text = self.extract_text_with_formatting(pdf_file)
                pairs = self.create_instruction_pairs(text, pdf_file.stem)
                all_pairs.extend(pairs)
                
            except Exception as e:
                print(f"Error processing {pdf_file.name}: {e}")
                continue
        
        # Save processed data
        output_file = self.output_dir / "training_data.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(all_pairs, f, indent=2, ensure_ascii=False)
        
        print(f"Processed {len(all_pairs)} instruction pairs")
        print(f"Saved to: {output_file}")
        
        return all_pairs

# Usage
if __name__ == "__main__":
    processor = PDFProcessor("/home/mlnerd/Desktop/DeepSeekCoderPDF/pdfs")
    training_data = processor.process_all_pdfs()
