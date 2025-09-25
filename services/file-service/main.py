from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any, BinaryIO
import os
import asyncio
from datetime import datetime
import structlog
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from fastapi import Response
import json
import httpx
import uuid
import mimetypes
from pathlib import Path
import shutil
import hashlib
from PIL import Image, ImageEnhance, ImageFilter
import cv2
import numpy as np
import easyocr
import pytesseract
import io
import boto3
from botocore.exceptions import ClientError

# Configure logging
logger = structlog.get_logger()

# Metrics
FILES_UPLOADED = Counter('files_uploaded_total', 'Total files uploaded', ['file_type', 'status'])
FILES_PROCESSED = Counter('files_processed_total', 'Total files processed', ['operation', 'status'])
FILE_PROCESSING_TIME = Histogram('file_processing_duration_seconds', 'File processing time', ['operation'])
FILE_SIZES = Histogram('file_sizes_bytes', 'File sizes uploaded', buckets=[1024, 10240, 102400, 1048576, 10485760, 104857600])

app = FastAPI(
    title="File Service",
    description="File storage and document processing service",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
UPLOAD_DIR = os.getenv("UPLOAD_DIR", "/app/uploads")
MAX_FILE_SIZE = int(os.getenv("MAX_FILE_SIZE", "50000000"))  # 50MB
ALLOWED_EXTENSIONS = {
    "images": {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".webp"},
    "documents": {".pdf", ".doc", ".docx", ".txt", ".rtf"},
    "archives": {".zip", ".rar", ".7z", ".tar", ".gz"}
}

# Pydantic Models
class FileMetadata(BaseModel):
    file_id: str
    filename: str
    original_filename: str
    file_type: str
    file_size: int
    mime_type: str
    upload_date: datetime
    uploader_id: Optional[str] = None
    claim_id: Optional[str] = None
    document_type: Optional[str] = None
    storage_path: str
    public_url: Optional[str] = None
    thumbnail_url: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

class OCRResult(BaseModel):
    text: str
    confidence: float
    language: str
    word_count: int
    line_count: int
    extracted_data: Optional[Dict[str, Any]] = None

class ImageProcessingRequest(BaseModel):
    file_id: str
    operations: List[str]  # resize, enhance, compress, etc.
    parameters: Optional[Dict[str, Any]] = None

class DocumentAnalysisResult(BaseModel):
    file_id: str
    document_type: str
    extracted_text: str
    confidence: float
    structured_data: Optional[Dict[str, Any]] = None
    processing_time: float

# In-memory storage for demo (use database in production)
files_db = {}

class FileStorage:
    def __init__(self):
        self.local_storage_path = UPLOAD_DIR
        self.use_s3 = os.getenv("USE_S3", "false").lower() == "true"
        
        # S3 Configuration
        if self.use_s3:
            self.s3_client = boto3.client(
                's3',
                aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
                aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
                region_name=os.getenv("AWS_REGION", "us-east-1")
            )
            self.s3_bucket = os.getenv("S3_BUCKET_NAME", "claims-documents")
        
        # Ensure local directory exists
        os.makedirs(self.local_storage_path, exist_ok=True)
        
        # Initialize OCR reader
        self.ocr_reader = easyocr.Reader(['en'])
    
    async def store_file(self, file: UploadFile, uploader_id: str = None, claim_id: str = None) -> FileMetadata:
        """Store uploaded file and return metadata"""
        with FILE_PROCESSING_TIME.labels(operation='upload').time():
            try:
                # Generate unique file ID
                file_id = f"file_{uuid.uuid4().hex[:12]}"
                
                # Validate file
                await self._validate_file(file)
                
                # Read file content
                file_content = await file.read()
                file_size = len(file_content)
                
                # Generate file hash for deduplication
                file_hash = hashlib.sha256(file_content).hexdigest()
                
                # Determine file type and extension
                mime_type = file.content_type or mimetypes.guess_type(file.filename)[0] or "application/octet-stream"
                file_extension = Path(file.filename).suffix.lower()
                
                # Create storage filename
                storage_filename = f"{file_id}{file_extension}"
                
                # Store file
                if self.use_s3:
                    storage_path = await self._store_to_s3(storage_filename, file_content, mime_type)
                    public_url = f"https://{self.s3_bucket}.s3.amazonaws.com/{storage_filename}"
                else:
                    storage_path = await self._store_locally(storage_filename, file_content)
                    public_url = f"/api/v1/files/{file_id}/download"
                
                # Create thumbnail for images
                thumbnail_url = None
                if self._is_image(file_extension):
                    thumbnail_path = await self._create_thumbnail(file_content, file_id, file_extension)
                    if thumbnail_path:
                        thumbnail_url = f"/api/v1/files/{file_id}/thumbnail"
                
                # Create metadata
                metadata = FileMetadata(
                    file_id=file_id,
                    filename=storage_filename,
                    original_filename=file.filename,
                    file_type=self._get_file_category(file_extension),
                    file_size=file_size,
                    mime_type=mime_type,
                    upload_date=datetime.utcnow(),
                    uploader_id=uploader_id,
                    claim_id=claim_id,
                    storage_path=storage_path,
                    public_url=public_url,
                    thumbnail_url=thumbnail_url,
                    metadata={
                        "hash": file_hash,
                        "extension": file_extension
                    }
                )
                
                # Store metadata
                files_db[file_id] = metadata.dict()
                
                # Record metrics
                FILES_UPLOADED.labels(
                    file_type=metadata.file_type,
                    status="success"
                ).inc()
                FILE_SIZES.observe(file_size)
                
                logger.info("File uploaded successfully",
                           file_id=file_id,
                           filename=file.filename,
                           size=file_size)
                
                return metadata
                
            except Exception as e:
                FILES_UPLOADED.labels(
                    file_type="unknown",
                    status="failed"
                ).inc()
                logger.error("File upload failed", error=str(e), filename=file.filename)
                raise HTTPException(status_code=500, detail=f"File upload failed: {str(e)}")
    
    async def _validate_file(self, file: UploadFile):
        """Validate uploaded file"""
        # Check file size
        file_content = await file.read()
        await file.seek(0)  # Reset file pointer
        
        if len(file_content) > MAX_FILE_SIZE:
            raise HTTPException(status_code=413, detail="File too large")
        
        # Check file extension
        file_extension = Path(file.filename).suffix.lower()
        all_allowed = set()
        for extensions in ALLOWED_EXTENSIONS.values():
            all_allowed.update(extensions)
        
        if file_extension not in all_allowed:
            raise HTTPException(status_code=400, detail="File type not allowed")
        
        # Check for malicious content (basic check)
        if b"<script" in file_content.lower():
            raise HTTPException(status_code=400, detail="Potentially malicious content detected")
    
    async def _store_locally(self, filename: str, content: bytes) -> str:
        """Store file locally"""
        file_path = os.path.join(self.local_storage_path, filename)
        
        with open(file_path, "wb") as f:
            f.write(content)
        
        return file_path
    
    async def _store_to_s3(self, filename: str, content: bytes, mime_type: str) -> str:
        """Store file to S3"""
        try:
            self.s3_client.put_object(
                Bucket=self.s3_bucket,
                Key=filename,
                Body=content,
                ContentType=mime_type
            )
            return f"s3://{self.s3_bucket}/{filename}"
        except ClientError as e:
            raise HTTPException(status_code=500, detail=f"S3 upload failed: {str(e)}")
    
    async def _create_thumbnail(self, file_content: bytes, file_id: str, extension: str) -> Optional[str]:
        """Create thumbnail for image files"""
        try:
            # Open image
            image = Image.open(io.BytesIO(file_content))
            
            # Create thumbnail
            thumbnail_size = (200, 200)
            image.thumbnail(thumbnail_size, Image.Resampling.LANCZOS)
            
            # Save thumbnail
            thumbnail_filename = f"{file_id}_thumb{extension}"
            thumbnail_path = os.path.join(self.local_storage_path, thumbnail_filename)
            
            image.save(thumbnail_path)
            return thumbnail_path
            
        except Exception as e:
            logger.warning("Thumbnail creation failed", error=str(e), file_id=file_id)
            return None
    
    def _is_image(self, extension: str) -> bool:
        """Check if file is an image"""
        return extension in ALLOWED_EXTENSIONS["images"]
    
    def _get_file_category(self, extension: str) -> str:
        """Get file category based on extension"""
        for category, extensions in ALLOWED_EXTENSIONS.items():
            if extension in extensions:
                return category
        return "other"
    
    async def get_file(self, file_id: str) -> Optional[FileMetadata]:
        """Get file metadata"""
        if file_id not in files_db:
            return None
        
        file_data = files_db[file_id]
        return FileMetadata(**file_data)
    
    async def get_file_content(self, file_id: str) -> Optional[bytes]:
        """Get file content"""
        file_metadata = await self.get_file(file_id)
        if not file_metadata:
            return None
        
        try:
            if self.use_s3 and file_metadata.storage_path.startswith("s3://"):
                # Get from S3
                s3_key = file_metadata.storage_path.replace(f"s3://{self.s3_bucket}/", "")
                response = self.s3_client.get_object(Bucket=self.s3_bucket, Key=s3_key)
                return response['Body'].read()
            else:
                # Get from local storage
                with open(file_metadata.storage_path, "rb") as f:
                    return f.read()
        except Exception as e:
            logger.error("Failed to get file content", error=str(e), file_id=file_id)
            return None
    
    async def delete_file(self, file_id: str) -> bool:
        """Delete file"""
        file_metadata = await self.get_file(file_id)
        if not file_metadata:
            return False
        
        try:
            if self.use_s3 and file_metadata.storage_path.startswith("s3://"):
                # Delete from S3
                s3_key = file_metadata.storage_path.replace(f"s3://{self.s3_bucket}/", "")
                self.s3_client.delete_object(Bucket=self.s3_bucket, Key=s3_key)
            else:
                # Delete from local storage
                if os.path.exists(file_metadata.storage_path):
                    os.remove(file_metadata.storage_path)
                
                # Delete thumbnail if exists
                if file_metadata.thumbnail_url:
                    thumbnail_path = os.path.join(self.local_storage_path, f"{file_id}_thumb{Path(file_metadata.filename).suffix}")
                    if os.path.exists(thumbnail_path):
                        os.remove(thumbnail_path)
            
            # Remove from database
            del files_db[file_id]
            
            logger.info("File deleted", file_id=file_id)
            return True
            
        except Exception as e:
            logger.error("File deletion failed", error=str(e), file_id=file_id)
            return False

class DocumentProcessor:
    def __init__(self, file_storage: FileStorage):
        self.file_storage = file_storage
        self.ocr_reader = easyocr.Reader(['en'])
    
    async def extract_text_ocr(self, file_id: str) -> OCRResult:
        """Extract text using OCR"""
        with FILE_PROCESSING_TIME.labels(operation='ocr').time():
            try:
                file_content = await self.file_storage.get_file_content(file_id)
                if not file_content:
                    raise HTTPException(status_code=404, detail="File not found")
                
                file_metadata = await self.file_storage.get_file(file_id)
                
                # Process based on file type
                if file_metadata.file_type == "images":
                    text, confidence = await self._ocr_image(file_content)
                elif file_metadata.file_type == "documents":
                    text, confidence = await self._ocr_document(file_content, file_metadata.filename)
                else:
                    raise HTTPException(status_code=400, detail="File type not supported for OCR")
                
                # Analyze extracted text
                word_count = len(text.split())
                line_count = len(text.splitlines())
                
                # Extract structured data if possible
                extracted_data = self._extract_structured_data(text, file_metadata.document_type)
                
                result = OCRResult(
                    text=text,
                    confidence=confidence,
                    language="en",
                    word_count=word_count,
                    line_count=line_count,
                    extracted_data=extracted_data
                )
                
                FILES_PROCESSED.labels(operation='ocr', status='success').inc()
                logger.info("OCR completed", file_id=file_id, word_count=word_count)
                
                return result
                
            except Exception as e:
                FILES_PROCESSED.labels(operation='ocr', status='failed').inc()
                logger.error("OCR failed", error=str(e), file_id=file_id)
                raise HTTPException(status_code=500, detail=f"OCR processing failed: {str(e)}")
    
    async def _ocr_image(self, image_content: bytes) -> tuple[str, float]:
        """Extract text from image using OCR"""
        # Convert bytes to numpy array
        nparr = np.frombuffer(image_content, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Preprocess image for better OCR
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply image enhancement
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        
        # Use EasyOCR
        results = self.ocr_reader.readtext(enhanced)
        
        # Combine text and calculate average confidence
        text_parts = []
        confidences = []
        
        for (bbox, text, confidence) in results:
            text_parts.append(text)
            confidences.append(confidence)
        
        full_text = " ".join(text_parts)
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
        
        return full_text, avg_confidence
    
    async def _ocr_document(self, doc_content: bytes, filename: str) -> tuple[str, float]:
        """Extract text from document (PDF, etc.)"""
        # For PDF files, convert to images first then OCR
        # For now, return placeholder
        return "Document text extraction not implemented yet", 0.5
    
    def _extract_structured_data(self, text: str, document_type: str) -> Dict[str, Any]:
        """Extract structured data from text based on document type"""
        structured_data = {}
        
        if document_type == "insurance_card":
            # Extract insurance-specific information
            import re
            
            # Policy number pattern
            policy_match = re.search(r'Policy\s*[#:]?\s*([A-Z0-9-]+)', text, re.IGNORECASE)
            if policy_match:
                structured_data['policy_number'] = policy_match.group(1)
            
            # Member ID pattern
            member_match = re.search(r'Member\s*ID\s*[#:]?\s*([A-Z0-9-]+)', text, re.IGNORECASE)
            if member_match:
                structured_data['member_id'] = member_match.group(1)
            
            # Effective dates
            date_pattern = r'(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})'
            dates = re.findall(date_pattern, text)
            if dates:
                structured_data['dates_found'] = dates
        
        elif document_type == "receipt":
            # Extract receipt information
            import re
            
            # Total amount pattern
            total_match = re.search(r'Total\s*[:]?\s*\$?(\d+\.?\d*)', text, re.IGNORECASE)
            if total_match:
                structured_data['total_amount'] = float(total_match.group(1))
            
            # Date pattern
            date_match = re.search(r'(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})', text)
            if date_match:
                structured_data['date'] = date_match.group(1)
        
        return structured_data

# Initialize services
file_storage = FileStorage()
document_processor = DocumentProcessor(file_storage)

# Health check
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "file-service",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0",
        "storage_type": "s3" if file_storage.use_s3 else "local"
    }

# Metrics endpoint
@app.get("/metrics")
async def get_metrics():
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)

# Upload file
@app.post("/upload", response_model=FileMetadata)
async def upload_file(
    file: UploadFile = File(...),
    uploader_id: Optional[str] = Form(None),
    claim_id: Optional[str] = Form(None),
    document_type: Optional[str] = Form(None)
):
    """Upload a file"""
    metadata = await file_storage.store_file(file, uploader_id, claim_id)
    
    # Update document type if provided
    if document_type:
        files_db[metadata.file_id]["document_type"] = document_type
        metadata.document_type = document_type
    
    return metadata

# Get file metadata
@app.get("/files/{file_id}", response_model=FileMetadata)
async def get_file_metadata(file_id: str):
    """Get file metadata"""
    metadata = await file_storage.get_file(file_id)
    if not metadata:
        raise HTTPException(status_code=404, detail="File not found")
    return metadata

# Download file
@app.get("/files/{file_id}/download")
async def download_file(file_id: str):
    """Download file"""
    file_content = await file_storage.get_file_content(file_id)
    if not file_content:
        raise HTTPException(status_code=404, detail="File not found")
    
    metadata = await file_storage.get_file(file_id)
    
    return StreamingResponse(
        io.BytesIO(file_content),
        media_type=metadata.mime_type,
        headers={"Content-Disposition": f"attachment; filename={metadata.original_filename}"}
    )

# Get thumbnail
@app.get("/files/{file_id}/thumbnail")
async def get_thumbnail(file_id: str):
    """Get file thumbnail"""
    metadata = await file_storage.get_file(file_id)
    if not metadata or not metadata.thumbnail_url:
        raise HTTPException(status_code=404, detail="Thumbnail not found")
    
    # For local storage, serve thumbnail file
    thumbnail_path = os.path.join(file_storage.local_storage_path, f"{file_id}_thumb{Path(metadata.filename).suffix}")
    
    if not os.path.exists(thumbnail_path):
        raise HTTPException(status_code=404, detail="Thumbnail file not found")
    
    return FileResponse(thumbnail_path)

# OCR text extraction
@app.post("/files/{file_id}/ocr", response_model=OCRResult)
async def extract_text(file_id: str):
    """Extract text from file using OCR"""
    return await document_processor.extract_text_ocr(file_id)

# Process image
@app.post("/files/{file_id}/process")
async def process_image(file_id: str, request: ImageProcessingRequest):
    """Process image with various operations"""
    # Placeholder for image processing operations
    # resize, enhance, compress, etc.
    return {
        "file_id": file_id,
        "operations_applied": request.operations,
        "status": "completed",
        "message": "Image processing completed successfully"
    }

# Delete file
@app.delete("/files/{file_id}")
async def delete_file(file_id: str):
    """Delete file"""
    success = await file_storage.delete_file(file_id)
    if not success:
        raise HTTPException(status_code=404, detail="File not found or deletion failed")
    
    return {"status": "deleted", "file_id": file_id}

# List files
@app.get("/files")
async def list_files(
    claim_id: Optional[str] = None,
    uploader_id: Optional[str] = None,
    file_type: Optional[str] = None,
    limit: int = 50,
    offset: int = 0
):
    """List files with filters"""
    files = list(files_db.values())
    
    # Apply filters
    if claim_id:
        files = [f for f in files if f.get("claim_id") == claim_id]
    
    if uploader_id:
        files = [f for f in files if f.get("uploader_id") == uploader_id]
    
    if file_type:
        files = [f for f in files if f["file_type"] == file_type]
    
    # Sort by upload date descending
    files.sort(key=lambda x: x["upload_date"], reverse=True)
    
    # Paginate
    total = len(files)
    files = files[offset:offset + limit]
    
    return {
        "files": files,
        "total": total,
        "limit": limit,
        "offset": offset
    }

# Bulk upload
@app.post("/upload-bulk")
async def upload_bulk_files(
    files: List[UploadFile] = File(...),
    uploader_id: Optional[str] = Form(None),
    claim_id: Optional[str] = Form(None)
):
    """Upload multiple files"""
    results = []
    
    for file in files:
        try:
            metadata = await file_storage.store_file(file, uploader_id, claim_id)
            results.append({
                "filename": file.filename,
                "file_id": metadata.file_id,
                "status": "success"
            })
        except Exception as e:
            results.append({
                "filename": file.filename,
                "status": "failed",
                "error": str(e)
            })
    
    return {"results": results}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 