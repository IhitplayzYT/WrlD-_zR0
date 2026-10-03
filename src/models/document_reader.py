from dataclasses import dataclass, field
from typing import Dict, Optional, Any 
from enum import Enum
import hashlib
from datetime import datetime
import re

class DocumentType(Enum):
    PDF = "pdf"
    IMAGE = "image"
    TEXT = "text"
    HTML = "html"
    MARKDOWN = "markdown"
    JSON = "json"

@dataclass
class Document:
    document_id: str
    document_type: DocumentType
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    source_path: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    checksum: str = ""
    
    def __post_init__(self):
        if not self.checksum:
            self.checksum = self._calculate_checksum()
    
    def _calculate_checksum(self) -> str:
        return hashlib.md5(self.content.encode()).hexdigest()
    
    def is_different_from(self, other: 'Document') -> bool:
        return self.checksum != other.checksum


class DocumentPreprocessor:
    def __init__(self):
        self.cleaning_rules = {'remove_extra_whitespace': True,'normalize_quotes': True,'remove_special_chars': False,'lowercase': False}
    
    def clean_text(self, text: str) -> str:
        if not text:
            return ""
        cleaned = text
        if self.cleaning_rules['remove_extra_whitespace']:
            cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        if self.cleaning_rules['normalize_quotes']:
            quote_map = str.maketrans({'“': '"','”': '"','„': '"','‟': '"','‘': "'",'’': "'",'‚': "'",'‛': "'"})
            cleaned = cleaned.translate(quote_map)
        if self.cleaning_rules['remove_special_chars']:
            cleaned = re.sub(r'[^\w\s\.\,\!\?\-\:\;]', '', cleaned)
        if self.cleaning_rules['lowercase']:
            cleaned = cleaned.lower()

        return cleaned
    
    def extract_key_information(self, text: str) -> Dict[str, Any]:
        return {'word_count': len(text.split()),'char_count': len(text),'sentence_count': len(re.split(r'[.!?]+', text)),'paragraph_count': len(re.split(r'\n\n+', text)),'has_numbers': bool(re.search(r'\d', text)),'has_urls': bool(re.search(r'http[s]?://', text)),'has_emails': bool(re.search(r'[\w\.-]+@[\w\.-]+', text)),'dates': re.findall(r'\d{4}-\d{2}-\d{2}|\d{2}/\d{2}/\d{4}', text),'numbers': re.findall(r'\d+\.?\d*', text)}        
    
    def summarize_document(self, text: str, max_length: int = 200) -> str:
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        if not sentences:
            return ""
        scored = []
        for i, sentence in enumerate(sentences):
            score = len(sentence.split())
            if i == 0 or i == len(sentences) - 1:
                score *= 1.5
            scored.append((score, sentence))
        
        scored.sort(reverse=True, key=lambda x: x[0])
        summary_sentences = [s for score, s in scored[:3]]
        summary = " ".join(summary_sentences)
        if len(summary) > max_length: # Truncating 
            summary = summary[:max_length//2 - 3] + "..." + summary[max_length//2:] 
        return summary[:max_length]
