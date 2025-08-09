"""
OCR (Optical Character Recognition) processing for the Research File Manager.

This module provides comprehensive OCR capabilities using EasyOCR for extracting
text from images and scanned PDF documents with confidence scoring and language detection.
"""

import os
import logging
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import json
import asyncio
import concurrent.futures
from dataclasses import dataclass

# OCR and image processing imports
try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

try:
    from pdf2image import convert_from_path
    PDF2IMAGE_AVAILABLE = True
except ImportError:
    PDF2IMAGE_AVAILABLE = False

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class OCRResult:
    """Result of OCR processing for a single image or document."""
    text: str
    confidence: float
    language_detected: str
    processing_time: float
    word_count: int
    error: Optional[str] = None
    metadata: Optional[Dict] = None


@dataclass
class OCRBatch:
    """Result of batch OCR processing."""
    results: List[OCRResult]
    total_text: str
    avg_confidence: float
    total_processing_time: float
    files_processed: int
    files_failed: int


class OCRProcessor:
    """
    Handles OCR processing for images and PDF documents using EasyOCR.
    
    Features:
    - Support for multiple image formats (.png, .jpg, .jpeg, .gif, .bmp)
    - PDF to image conversion for scanned PDFs
    - Configurable confidence thresholds
    - Language detection and multi-language support
    - Batch processing capabilities
    - Memory-efficient processing for large files
    - Progress tracking for long operations
    """
    
    # Supported image formats
    IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff', '.tif', '.webp'}
    
    # PDF extension
    PDF_EXTENSIONS = {'.pdf'}
    
    # Supported file types for OCR
    OCR_EXTENSIONS = IMAGE_EXTENSIONS | PDF_EXTENSIONS
    
    # Maximum file sizes for processing (in bytes)
    MAX_IMAGE_SIZE = 50 * 1024 * 1024  # 50MB
    MAX_PDF_SIZE = 200 * 1024 * 1024   # 200MB
    
    # OCR configuration defaults
    DEFAULT_LANGUAGES = ['en']  # English by default
    DEFAULT_CONFIDENCE_THRESHOLD = 0.5
    DEFAULT_MAX_PAGES = 20  # For PDF processing
    
    def __init__(self, 
                 languages: List[str] = None,
                 confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
                 gpu: bool = True,
                 download_enabled: bool = True):
        """
        Initialize the OCR processor.
        
        Args:
            languages: List of language codes for OCR (e.g., ['en', 'es', 'fr'])
            confidence_threshold: Minimum confidence score for OCR results
            gpu: Whether to use GPU acceleration if available
            download_enabled: Whether to allow model downloads
        """
        self.languages = languages or self.DEFAULT_LANGUAGES
        self.confidence_threshold = confidence_threshold
        self.gpu = gpu
        self.download_enabled = download_enabled
        
        # Initialize EasyOCR reader
        self.reader = None
        self._init_reader()
        
        # Thread pool for async processing
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)
        
        logger.info(f"OCRProcessor initialized with languages: {self.languages}, "
                   f"GPU: {self.gpu}, confidence_threshold: {self.confidence_threshold}")
    
    def _init_reader(self) -> None:
        """Initialize the EasyOCR reader with error handling."""
        if not EASYOCR_AVAILABLE:
            logger.error("EasyOCR not available. Install with: pip install easyocr")
            return
        
        try:
            logger.info("Initializing EasyOCR reader...")
            self.reader = easyocr.Reader(
                self.languages,
                gpu=self.gpu,
                download_enabled=self.download_enabled,
                verbose=False
            )
            logger.info("✅ EasyOCR reader initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize EasyOCR reader: {e}")
            self.reader = None
    
    @property
    def is_available(self) -> bool:
        """Check if OCR processing is available."""
        return (EASYOCR_AVAILABLE and PIL_AVAILABLE and 
                self.reader is not None)
    
    @property
    def pdf_support_available(self) -> bool:
        """Check if PDF OCR processing is available."""
        return self.is_available and PDF2IMAGE_AVAILABLE
    
    def get_capabilities(self) -> Dict:
        """Get the current OCR capabilities and status."""
        return {
            'ocr_available': self.is_available,
            'pdf_support': self.pdf_support_available,
            'supported_languages': self.languages,
            'supported_image_formats': list(self.IMAGE_EXTENSIONS),
            'supported_pdf': self.pdf_support_available,
            'confidence_threshold': self.confidence_threshold,
            'max_image_size_mb': self.MAX_IMAGE_SIZE // (1024 * 1024),
            'max_pdf_size_mb': self.MAX_PDF_SIZE // (1024 * 1024),
            'dependencies': {
                'easyocr': EASYOCR_AVAILABLE,
                'pillow': PIL_AVAILABLE,
                'pdf2image': PDF2IMAGE_AVAILABLE
            }
        }
    
    def can_process_file(self, file_path: str) -> bool:
        """
        Check if a file can be processed with OCR.
        
        Args:
            file_path: Path to the file
            
        Returns:
            True if file can be processed, False otherwise
        """
        if not self.is_available:
            return False
        
        try:
            file_path_obj = Path(file_path)
            file_ext = file_path_obj.suffix.lower()
            
            # Check if file extension is supported
            if file_ext not in self.OCR_EXTENSIONS:
                return False
            
            # Check file size limits
            file_size = os.path.getsize(file_path)
            if file_ext in self.IMAGE_EXTENSIONS and file_size > self.MAX_IMAGE_SIZE:
                return False
            elif file_ext in self.PDF_EXTENSIONS and file_size > self.MAX_PDF_SIZE:
                return False
            
            # For PDFs, check if PDF support is available
            if file_ext in self.PDF_EXTENSIONS and not self.pdf_support_available:
                return False
            
            return True
            
        except (OSError, IOError):
            return False
    
    async def extract_text_async(self, file_path: str, 
                                max_pages: int = None) -> OCRResult:
        """
        Asynchronously extract text from an image or PDF file.
        
        Args:
            file_path: Path to the file
            max_pages: Maximum pages to process for PDFs
            
        Returns:
            OCRResult with extracted text and metadata
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor,
            self.extract_text,
            file_path,
            max_pages
        )
    
    def extract_text(self, file_path: str, max_pages: int = None) -> OCRResult:
        """
        Extract text from an image or PDF file using OCR.
        
        Args:
            file_path: Path to the file
            max_pages: Maximum pages to process for PDFs (default: DEFAULT_MAX_PAGES)
            
        Returns:
            OCRResult with extracted text and metadata
        """
        start_time = datetime.now()
        
        if not self.can_process_file(file_path):
            return OCRResult(
                text="",
                confidence=0.0,
                language_detected="unknown",
                processing_time=0.0,
                word_count=0,
                error=f"File cannot be processed: {file_path}"
            )
        
        try:
            file_path_obj = Path(file_path)
            file_ext = file_path_obj.suffix.lower()
            
            if file_ext in self.IMAGE_EXTENSIONS:
                result = self._process_image(file_path)
            elif file_ext in self.PDF_EXTENSIONS:
                max_pages = max_pages or self.DEFAULT_MAX_PAGES
                result = self._process_pdf(file_path, max_pages)
            else:
                raise ValueError(f"Unsupported file type: {file_ext}")
            
            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds()
            result.processing_time = processing_time
            
            logger.info(f"OCR completed for {file_path}: {len(result.text)} chars, "
                       f"confidence: {result.confidence:.2f}, time: {processing_time:.2f}s")
            
            return result
            
        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds()
            logger.error(f"OCR failed for {file_path}: {e}")
            return OCRResult(
                text="",
                confidence=0.0,
                language_detected="unknown",
                processing_time=processing_time,
                word_count=0,
                error=str(e)
            )
    
    def _process_image(self, image_path: str) -> OCRResult:
        """Process a single image file with OCR."""
        try:
            # Load and validate image
            with Image.open(image_path) as img:
                # Convert to RGB if necessary
                if img.mode not in ('RGB', 'L'):
                    img = img.convert('RGB')
                
                # Check image dimensions (prevent memory issues)
                width, height = img.size
                max_dimension = 4000  # Maximum dimension in pixels
                if width > max_dimension or height > max_dimension:
                    # Resize image to manageable size
                    ratio = min(max_dimension / width, max_dimension / height)
                    new_size = (int(width * ratio), int(height * ratio))
                    img = img.resize(new_size, Image.Resampling.LANCZOS)
                    logger.info(f"Resized image from {width}x{height} to {new_size[0]}x{new_size[1]}")
                
                # Perform OCR
                results = self.reader.readtext(image_path, detail=1)
                
                # Process OCR results
                return self._process_ocr_results(results, {"image_size": f"{width}x{height}"})
                
        except Exception as e:
            raise Exception(f"Image processing failed: {e}")
    
    def _process_pdf(self, pdf_path: str, max_pages: int) -> OCRResult:
        """Process a PDF file by converting to images and running OCR."""
        if not PDF2IMAGE_AVAILABLE:
            raise Exception("PDF processing requires pdf2image. Install with: pip install pdf2image")
        
        try:
            # Convert PDF to images
            logger.info(f"Converting PDF to images: {pdf_path}")
            images = convert_from_path(
                pdf_path,
                first_page=1,
                last_page=max_pages,
                dpi=200,  # Good balance of quality and speed
                fmt='RGB'
            )
            
            if not images:
                raise Exception("No pages found in PDF")
            
            logger.info(f"Processing {len(images)} pages from PDF")
            
            # Process each page
            all_results = []
            for i, image in enumerate(images):
                try:
                    # Convert PIL image to format for EasyOCR
                    import numpy as np
                    image_array = np.array(image)
                    
                    # Perform OCR on this page
                    page_results = self.reader.readtext(image_array, detail=1)
                    all_results.extend(page_results)
                    
                except Exception as e:
                    logger.warning(f"Failed to process PDF page {i+1}: {e}")
                    continue
            
            # Combine results from all pages
            metadata = {
                "pdf_pages": len(images),
                "pages_processed": len([r for r in all_results if r])
            }
            
            return self._process_ocr_results(all_results, metadata)
            
        except Exception as e:
            raise Exception(f"PDF processing failed: {e}")
    
    def _process_ocr_results(self, ocr_results: List, metadata: Dict = None) -> OCRResult:
        """
        Process raw OCR results from EasyOCR into structured format.
        
        Args:
            ocr_results: Raw results from EasyOCR readtext
            metadata: Additional metadata to include
            
        Returns:
            Processed OCRResult
        """
        if not ocr_results:
            return OCRResult(
                text="",
                confidence=0.0,
                language_detected="unknown",
                processing_time=0.0,
                word_count=0,
                metadata=metadata or {}
            )
        
        # Extract text and confidences
        texts = []
        confidences = []
        
        for result in ocr_results:
            if len(result) >= 3:  # [bbox, text, confidence]
                text = result[1].strip()
                confidence = float(result[2])
                
                # Apply confidence threshold
                if confidence >= self.confidence_threshold:
                    texts.append(text)
                    confidences.append(confidence)
        
        # Combine text
        full_text = '\n'.join(texts)
        
        # Calculate statistics
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
        word_count = len(full_text.split()) if full_text else 0
        
        # Simple language detection based on character patterns
        detected_language = self._detect_language(full_text)
        
        # Combine metadata
        result_metadata = metadata or {}
        result_metadata.update({
            'total_detections': len(ocr_results),
            'filtered_detections': len(texts),
            'confidence_threshold_applied': self.confidence_threshold,
            'languages_configured': self.languages
        })
        
        return OCRResult(
            text=full_text,
            confidence=avg_confidence,
            language_detected=detected_language,
            processing_time=0.0,  # Will be set by caller
            word_count=word_count,
            metadata=result_metadata
        )
    
    def _detect_language(self, text: str) -> str:
        """
        Simple language detection based on character patterns.
        
        Args:
            text: Text to analyze
            
        Returns:
            Detected language code
        """
        if not text:
            return "unknown"
        
        # Count different character types
        ascii_chars = sum(1 for c in text if ord(c) < 128)
        total_chars = len(text)
        
        if total_chars == 0:
            return "unknown"
        
        ascii_ratio = ascii_chars / total_chars
        
        # Simple heuristic: if mostly ASCII, likely English or similar
        if ascii_ratio > 0.8:
            return "en"
        else:
            # Return the first configured language that's not English
            non_english = [lang for lang in self.languages if lang != 'en']
            return non_english[0] if non_english else "unknown"
    
    async def process_batch(self, file_paths: List[str], 
                           max_pages_per_pdf: int = None) -> OCRBatch:
        """
        Process multiple files in batch.
        
        Args:
            file_paths: List of file paths to process
            max_pages_per_pdf: Maximum pages per PDF
            
        Returns:
            OCRBatch with combined results
        """
        start_time = datetime.now()
        
        # Create async tasks for each file
        tasks = [
            self.extract_text_async(file_path, max_pages_per_pdf)
            for file_path in file_paths
            if self.can_process_file(file_path)
        ]
        
        if not tasks:
            return OCRBatch(
                results=[],
                total_text="",
                avg_confidence=0.0,
                total_processing_time=0.0,
                files_processed=0,
                files_failed=0
            )
        
        # Process all files concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Separate successful results from failures
        successful_results = []
        failed_count = 0
        
        for result in results:
            if isinstance(result, OCRResult):
                if result.error:
                    failed_count += 1
                else:
                    successful_results.append(result)
            else:
                failed_count += 1
        
        # Combine results
        total_text = '\n\n'.join(r.text for r in successful_results if r.text)
        avg_confidence = (sum(r.confidence for r in successful_results) / 
                         len(successful_results)) if successful_results else 0.0
        
        total_processing_time = (datetime.now() - start_time).total_seconds()
        
        return OCRBatch(
            results=successful_results,
            total_text=total_text,
            avg_confidence=avg_confidence,
            total_processing_time=total_processing_time,
            files_processed=len(successful_results),
            files_failed=failed_count
        )
    
    def get_file_ocr_status(self, file_path: str) -> Dict:
        """
        Get OCR processing status and metadata for a file.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Dictionary with OCR status information
        """
        file_path_obj = Path(file_path)
        file_ext = file_path_obj.suffix.lower()
        
        try:
            file_size = os.path.getsize(file_path)
        except (OSError, IOError):
            file_size = 0
        
        status = {
            'file_path': file_path,
            'file_extension': file_ext,
            'file_size': file_size,
            'can_process': self.can_process_file(file_path),
            'is_image': file_ext in self.IMAGE_EXTENSIONS,
            'is_pdf': file_ext in self.PDF_EXTENSIONS,
            'ocr_available': self.is_available,
            'estimated_processing_time': self._estimate_processing_time(file_path),
        }
        
        if not status['can_process']:
            status['reason'] = self._get_cannot_process_reason(file_path)
        
        return status
    
    def _estimate_processing_time(self, file_path: str) -> float:
        """
        Estimate processing time for a file based on size and type.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Estimated processing time in seconds
        """
        try:
            file_size = os.path.getsize(file_path)
            file_ext = Path(file_path).suffix.lower()
            
            # Base processing time estimates (seconds per MB)
            if file_ext in self.IMAGE_EXTENSIONS:
                base_time_per_mb = 2.0  # 2 seconds per MB for images
            elif file_ext in self.PDF_EXTENSIONS:
                base_time_per_mb = 5.0  # 5 seconds per MB for PDFs
            else:
                return 0.0
            
            file_size_mb = file_size / (1024 * 1024)
            return file_size_mb * base_time_per_mb
            
        except (OSError, IOError):
            return 0.0
    
    def _get_cannot_process_reason(self, file_path: str) -> str:
        """Get reason why a file cannot be processed."""
        if not self.is_available:
            return "OCR service not available (missing dependencies)"
        
        try:
            file_path_obj = Path(file_path)
            file_ext = file_path_obj.suffix.lower()
            file_size = os.path.getsize(file_path)
            
            if file_ext not in self.OCR_EXTENSIONS:
                return f"Unsupported file type: {file_ext}"
            
            if file_ext in self.IMAGE_EXTENSIONS and file_size > self.MAX_IMAGE_SIZE:
                return f"Image file too large: {file_size / (1024*1024):.1f}MB (max: {self.MAX_IMAGE_SIZE / (1024*1024):.1f}MB)"
            
            if file_ext in self.PDF_EXTENSIONS:
                if not self.pdf_support_available:
                    return "PDF processing not available (missing pdf2image)"
                if file_size > self.MAX_PDF_SIZE:
                    return f"PDF file too large: {file_size / (1024*1024):.1f}MB (max: {self.MAX_PDF_SIZE / (1024*1024):.1f}MB)"
            
            return "Unknown reason"
            
        except (OSError, IOError):
            return "File not accessible"
    
    def __del__(self):
        """Cleanup resources."""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=False)


# Global OCR processor instance
_ocr_processor = None


def get_ocr_processor() -> Optional[OCRProcessor]:
    """Get the global OCR processor instance."""
    global _ocr_processor
    
    if _ocr_processor is None:
        try:
            _ocr_processor = OCRProcessor()
        except Exception as e:
            logger.error(f"Failed to initialize OCR processor: {e}")
            return None
    
    return _ocr_processor


def is_ocr_available() -> bool:
    """Check if OCR processing is available."""
    processor = get_ocr_processor()
    return processor is not None and processor.is_available


def get_ocr_capabilities() -> Dict:
    """Get OCR capabilities and status."""
    processor = get_ocr_processor()
    if processor:
        return processor.get_capabilities()
    else:
        return {
            'ocr_available': False,
            'error': 'OCR processor not initialized'
        }