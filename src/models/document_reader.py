# For better parsing of Financial Documents

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union
from enum import Enum
from pathlib import Path
import hashlib
import json
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
    """
    Document representation with metadata and content.
    """
    document_id: str
    document_type: DocumentType
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    source_path: Optional[str] = None
    version: str = "1.0"
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    checksum: str = ""
    
    def __post_init__(self):
        """Calculate checksum after initialization"""
        if not self.checksum:
            self.checksum = self._calculate_checksum()
    
    def _calculate_checksum(self) -> str:
        """Calculate MD5 checksum of content"""
        return hashlib.md5(self.content.encode()).hexdigest()
    
    def is_different_from(self, other: 'Document') -> bool:
        """Check if this document is different from another"""
        return self.checksum != other.checksum


class DocumentPreprocessor:
    
    def __init__(self):
        self.cleaning_rules = {'remove_extra_whitespace': True,'normalize_quotes': True,'remove_special_chars': False,'lowercase': False}
    
    def clean_text(self, text: str) -> str:
        if not text:
            return ""
        cleaned = text
        
        if self.cleaning_rules['remove_extra_whitespace']:
            cleaned = re.sub(r'\s+', ' ', cleaned)
            cleaned = cleaned.strip()
        
        if self.cleaning_rules['normalize_quotes']:
            cleaned = cleaned.replace('"', '"').replace('"', '"')
            cleaned = cleaned.replace(''', "'").replace(''', "'")
        
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
            # The first and last sentencew usually give intro and final conclusion about the topic 
            if i == 0 or i == len(sentences) - 1:
                score *= 1.5
            scored.append((score, sentence))
        
        scored.sort(reverse=True, key=lambda x: x[0])
        summary_sentences = [s for score, s in scored[:3]]
        summary = " ".join(summary_sentences)
        if len(summary) > max_length:
            summary = summary[:max_length//2 - 3] + "..." + summary[max_length//2:]
        return summary[:max_length]


class DocumentReader:
    
    def __init__(self, cache_dir: str = "./document_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.preprocessor = DocumentPreprocessor()
        self.document_cache: Dict[str, Document] = {}
        self.version_history: Dict[str, List[Document]] = {}
    
    def read_document(self,file_path: str,force_reload: bool = False) -> Document:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Document not found: {file_path}")
        
        doc_id = self._generate_document_id(file_path)
        if not force_reload and doc_id in self.document_cache:
            return self.document_cache[doc_id]
        
        doc_type = self._detect_document_type(path)
        content = self._read_content(path, doc_type)
        cleaned_content = self.preprocessor.clean_text(content)
        metadata = {'file_path': str(path),'file_size': path.stat().st_size,'file_extension': path.suffix,'key_info': self.preprocessor.extract_key_information(cleaned_content),'summary': self.preprocessor.summarize_document(cleaned_content)}
        document = Document(document_id=doc_id,document_type=doc_type,content=cleaned_content,metadata=metadata,source_path=str(path))
        self._cache_document(document)
        return document
    
    def _generate_document_id(self, file_path: str) -> str:
        path = Path(file_path)
        return hashlib.md5(str(path.absolute()).encode()).hexdigest()
    
    def _detect_document_type(self, path: Path) -> DocumentType:
        extension = path.suffix.lower()
        type_mapping = {
            '.pdf': DocumentType.PDF,
            '.png': DocumentType.IMAGE,
            '.jpg': DocumentType.IMAGE,
            '.jpeg': DocumentType.IMAGE,
            '.gif': DocumentType.IMAGE,
            '.txt': DocumentType.TEXT,
            '.html': DocumentType.HTML,
            '.htm': DocumentType.HTML,
            '.md': DocumentType.MARKDOWN,
            '.json': DocumentType.JSON
        }
        
        return type_mapping.get(extension, DocumentType.TEXT)
    
    def _read_content(self, path: Path, doc_type: DocumentType) -> str:
        if doc_type == DocumentType.TEXT:
            return path.read_text(encoding='utf-8')
        elif doc_type == DocumentType.JSON:
            data = json.loads(path.read_text(encoding='utf-8'))
            return json.dumps(data, indent=2)
        elif doc_type == DocumentType.MARKDOWN:
            return path.read_text(encoding='utf-8')
        elif doc_type == DocumentType.HTML:
            return path.read_text(encoding='utf-8')
        elif doc_type == DocumentType.PDF:
            try:
                import pymupdf
                doc = pymupdf.open(path)
                text = ""
                for page in doc:
                    text += page.get_text()
                doc.close()
                return text
            except ImportError:
                return f"[PDF content not available - pymupdf not installed]"
        
        elif doc_type == DocumentType.IMAGE:
            try:
                from PIL import Image
                import pytesseract
                
                image = Image.open(path)
                text = pytesseract.image_to_string(image)
                return text
            except ImportError:
                return f"[Image content not available - OCR not installed]"
        else:
            return path.read_text(encoding='utf-8', errors='ignore')
    
    def _cache_document(self, document: Document):
        self.document_cache[document.document_id] = document
        if document.document_id not in self.version_history:
            self.version_history[document.document_id] = []
        self.version_history[document.document_id].append(document)
        cache_file = self.cache_dir / f"{document.document_id}.json"
        cache_data = {'document_id': document.document_id,'document_type': document.document_type.value,'content': document.content,'metadata': document.metadata,'source_path': document.source_path,'version': document.version,'created_at': document.created_at.isoformat(),'updated_at': document.updated_at.isoformat(),'checksum': document.checksum}
        with open(cache_file, 'w') as f:
            json.dump(cache_data, f, indent=2)
    
    def load_from_cache(self, document_id: str) -> Optional[Document]:
        if document_id in self.document_cache:
            return self.document_cache[document_id]
        cache_file = self.cache_dir / f"{document_id}.json"
        if cache_file.exists():
            with open(cache_file, 'r') as f:
                cache_data = json.load(f)
            self.document_cache[document_id] = Document(document_id=cache_data['document_id'],document_type=DocumentType(cache_data['document_type']),content=cache_data['content'],metadata=cache_data['metadata'],source_path=cache_data['source_path'],version=cache_data['version'],created_at=datetime.fromisoformat(cache_data['created_at']),updated_at=datetime.fromisoformat(cache_data['updated_at']),checksum=cache_data['checksum'])
            return self.document_cache[document_id]     
        return None
    
    def detect_changes(self, file_path: str) -> bool:
        doc_id = self._generate_document_id(file_path)
        if doc_id not in self.document_cache:
            return True
        current_doc = self.read_document(file_path, force_reload=True)
        cached_doc = self.document_cache[doc_id]
        return current_doc.is_different_from(cached_doc)
    
    def get_document_versions(self, document_id: str) -> List[Document]:
        return self.version_history.get(document_id, [])
    
    def send_to_agents(self,document: Document,agent_ids: List[str],agent_message_bus: Optional[Any] = None):
        message = {'type': 'document_update','document_id': document.document_id,'document_type': document.document_type.value,'content': document.content,'summary': document.metadata.get('summary', ''),'key_info': document.metadata.get('key_info', {}),'timestamp': datetime.now().isoformat()}
        if agent_message_bus:
            for agent_id in agent_ids:
                agent_message_bus.send(agent_id, message)
        else:
            return message
    
    def batch_read_documents(self,file_paths: List[str],force_reload: bool = False) -> Dict[str, Document]:
        documents = {}
        for file_path in file_paths:
            try:
                document = self.read_document(file_path, force_reload)
                documents[document.document_id] = document
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
        return documents
    
    def clear_cache(self):
        self.document_cache.clear()
        self.version_history.clear()
        for cache_file in self.cache_dir.glob("*.json"):
            cache_file.unlink()


if __name__ == "__main__":
    reader = DocumentReader()
    
    test_file = Path("./test_document.txt")
    test_file.write_text("This is a test document for the RLAIF trading simulation. "
                        "It contains some sample text that agents can read and process.")
    
    document = reader.read_document(str(test_file))
    
    print(f"Document ID: {document.document_id}")
    print(f"Document Type: {document.document_type.value}")
    print(f"Content: {document.content}")
    print(f"Word Count: {document.metadata['key_info']['word_count']}")
    print(f"Summary: {document.metadata['summary']}")
    
    test_file.unlink()
    reader.clear_cache()
    
    print("Document reader test completed")
