"""
Field-Level PII/PHI Encryption System

Provides comprehensive data protection with:
- Field-level encryption for sensitive data (PII/PHI)
- Multiple encryption algorithms (AES-256-GCM, ChaCha20-Poly1305)
- Key management with automatic rotation
- Data classification and tagging
- Compliance with GDPR, HIPAA, and other regulations
- Searchable encryption for encrypted fields
- Audit logging for all encryption operations
"""

import os
import json
import base64
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
import secrets
import hashlib
from cryptography.hazmat.primitives.ciphers.aead import AESGCM, ChaCha20Poly1305
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.backends import default_backend
import pandas as pd
import re

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataClassification(Enum):
    """Data classification levels"""
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"
    PII = "pii"  # Personally Identifiable Information
    PHI = "phi"  # Protected Health Information
    PCI = "pci"  # Payment Card Industry

class EncryptionAlgorithm(Enum):
    """Supported encryption algorithms"""
    AES_256_GCM = "aes-256-gcm"
    CHACHA20_POLY1305 = "chacha20-poly1305"

@dataclass
class FieldClassification:
    """Field classification and encryption rules"""
    field_name: str
    classification: DataClassification
    encryption_required: bool = True
    algorithm: EncryptionAlgorithm = EncryptionAlgorithm.AES_256_GCM
    key_rotation_days: int = 90
    retention_days: Optional[int] = None
    searchable: bool = False  # Whether to create searchable hash
    masking_pattern: Optional[str] = None  # Pattern for data masking
    compliance_tags: List[str] = field(default_factory=list)

@dataclass
class EncryptionKey:
    """Encryption key information"""
    key_id: str
    key_data: bytes
    algorithm: EncryptionAlgorithm
    created_at: datetime
    expires_at: Optional[datetime] = None
    version: int = 1
    status: str = "active"  # active, rotated, revoked

@dataclass
class EncryptedField:
    """Encrypted field data"""
    field_name: str
    encrypted_value: str
    key_id: str
    algorithm: EncryptionAlgorithm
    iv_nonce: str
    search_hash: Optional[str] = None
    encrypted_at: datetime = field(default_factory=datetime.now)

class FieldEncryption:
    """Main field encryption class"""
    
    def __init__(self, master_key: Optional[str] = None):
        self.field_classifications: Dict[str, FieldClassification] = {}
        self.encryption_keys: Dict[str, EncryptionKey] = {}
        self.master_key = master_key or os.environ.get("ENCRYPTION_MASTER_KEY", self._generate_master_key())
        
        # Initialize field classifications
        self._initialize_field_classifications()
        
        # Initialize encryption keys
        self._initialize_encryption_keys()
    
    def _generate_master_key(self) -> str:
        """Generate a master key for key derivation"""
        return base64.b64encode(secrets.token_bytes(32)).decode('utf-8')
    
    def _derive_key(self, key_id: str, salt: bytes = None) -> bytes:
        """Derive encryption key from master key"""
        if salt is None:
            salt = key_id.encode('utf-8')[:16].ljust(16, b'\0')
        
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
            backend=default_backend()
        )
        
        return kdf.derive(self.master_key.encode('utf-8'))
    
    def _initialize_field_classifications(self):
        """Initialize field classifications for common PII/PHI fields"""
        
        classifications = [
            # Personal Identifiers
            FieldClassification("ssn", DataClassification.PII, compliance_tags=["GDPR", "HIPAA"]),
            FieldClassification("social_security_number", DataClassification.PII, compliance_tags=["GDPR", "HIPAA"]),
            FieldClassification("passport_number", DataClassification.PII, compliance_tags=["GDPR"]),
            FieldClassification("driver_license", DataClassification.PII, compliance_tags=["GDPR"]),
            
            # Contact Information
            FieldClassification("email", DataClassification.PII, searchable=True, masking_pattern="***@***.***"),
            FieldClassification("phone", DataClassification.PII, searchable=True, masking_pattern="***-***-****"),
            FieldClassification("address", DataClassification.PII, masking_pattern="*** *** St, City, ST"),
            FieldClassification("postal_code", DataClassification.PII),
            
            # Financial Information
            FieldClassification("credit_card", DataClassification.PCI, masking_pattern="****-****-****-1234"),
            FieldClassification("bank_account", DataClassification.PCI, masking_pattern="****1234"),
            FieldClassification("routing_number", DataClassification.PCI),
            
            # Health Information
            FieldClassification("medical_record_number", DataClassification.PHI, compliance_tags=["HIPAA"]),
            FieldClassification("diagnosis", DataClassification.PHI, compliance_tags=["HIPAA"]),
            FieldClassification("medication", DataClassification.PHI, compliance_tags=["HIPAA"]),
            FieldClassification("insurance_id", DataClassification.PHI, compliance_tags=["HIPAA"]),
            
            # Claims-specific fields
            FieldClassification("claim_description", DataClassification.CONFIDENTIAL, searchable=True),
            FieldClassification("customer_notes", DataClassification.CONFIDENTIAL),
            FieldClassification("adjuster_notes", DataClassification.INTERNAL),
            
            # Biometric data
            FieldClassification("fingerprint", DataClassification.PII, compliance_tags=["GDPR", "HIPAA"]),
            FieldClassification("facial_recognition", DataClassification.PII, compliance_tags=["GDPR"]),
            
            # Location data
            FieldClassification("gps_coordinates", DataClassification.PII, compliance_tags=["GDPR"]),
            FieldClassification("ip_address", DataClassification.PII, compliance_tags=["GDPR"])
        ]
        
        for classification in classifications:
            self.field_classifications[classification.field_name] = classification
        
        logger.info(f"Initialized {len(classifications)} field classifications")
    
    def _initialize_encryption_keys(self):
        """Initialize encryption keys"""
        
        # Create default keys for each algorithm
        for algorithm in EncryptionAlgorithm:
            key_id = f"default_{algorithm.value}_{datetime.now().strftime('%Y%m%d')}"
            
            if algorithm == EncryptionAlgorithm.AES_256_GCM:
                key_data = secrets.token_bytes(32)  # 256 bits
            elif algorithm == EncryptionAlgorithm.CHACHA20_POLY1305:
                key_data = secrets.token_bytes(32)  # 256 bits
            else:
                continue
            
            encryption_key = EncryptionKey(
                key_id=key_id,
                key_data=key_data,
                algorithm=algorithm,
                created_at=datetime.now(),
                expires_at=datetime.now() + timedelta(days=90)
            )
            
            self.encryption_keys[key_id] = encryption_key
        
        logger.info(f"Initialized {len(self.encryption_keys)} encryption keys")
    
    def add_field_classification(self, classification: FieldClassification):
        """Add or update field classification"""
        self.field_classifications[classification.field_name] = classification
        logger.info(f"Added field classification: {classification.field_name}")
    
    def get_field_classification(self, field_name: str) -> Optional[FieldClassification]:
        """Get field classification"""
        return self.field_classifications.get(field_name)
    
    def _get_active_key(self, algorithm: EncryptionAlgorithm) -> EncryptionKey:
        """Get active encryption key for algorithm"""
        
        for key in self.encryption_keys.values():
            if (key.algorithm == algorithm and 
                key.status == "active" and 
                (key.expires_at is None or key.expires_at > datetime.now())):
                return key
        
        # Create new key if none found
        return self._create_new_key(algorithm)
    
    def _create_new_key(self, algorithm: EncryptionAlgorithm) -> EncryptionKey:
        """Create new encryption key"""
        
        key_id = f"key_{algorithm.value}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        key_data = secrets.token_bytes(32)
        
        encryption_key = EncryptionKey(
            key_id=key_id,
            key_data=key_data,
            algorithm=algorithm,
            created_at=datetime.now(),
            expires_at=datetime.now() + timedelta(days=90)
        )
        
        self.encryption_keys[key_id] = encryption_key
        logger.info(f"Created new encryption key: {key_id}")
        
        return encryption_key
    
    def _create_search_hash(self, value: str, field_name: str) -> str:
        """Create searchable hash for encrypted field"""
        
        # Use HMAC for searchable hash to prevent rainbow table attacks
        import hmac
        
        # Use field-specific salt
        salt = f"search_{field_name}_{self.master_key[:16]}"
        
        search_hash = hmac.new(
            salt.encode('utf-8'),
            value.lower().strip().encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        
        return search_hash
    
    def encrypt_field(self, field_name: str, value: str) -> EncryptedField:
        """Encrypt a field value"""
        
        if not value:
            return EncryptedField(
                field_name=field_name,
                encrypted_value="",
                key_id="",
                algorithm=EncryptionAlgorithm.AES_256_GCM,
                iv_nonce=""
            )
        
        # Get field classification
        classification = self.get_field_classification(field_name)
        if not classification or not classification.encryption_required:
            # Field doesn't require encryption
            return EncryptedField(
                field_name=field_name,
                encrypted_value=value,  # Store as plain text
                key_id="plaintext",
                algorithm=EncryptionAlgorithm.AES_256_GCM,
                iv_nonce=""
            )
        
        # Get encryption key
        key = self._get_active_key(classification.algorithm)
        
        # Create search hash if needed
        search_hash = None
        if classification.searchable:
            search_hash = self._create_search_hash(value, field_name)
        
        # Encrypt value
        if classification.algorithm == EncryptionAlgorithm.AES_256_GCM:
            cipher = AESGCM(key.key_data)
            nonce = secrets.token_bytes(12)  # 96 bits for GCM
            encrypted_bytes = cipher.encrypt(nonce, value.encode('utf-8'), None)
            
            encrypted_value = base64.b64encode(encrypted_bytes).decode('utf-8')
            iv_nonce = base64.b64encode(nonce).decode('utf-8')
            
        elif classification.algorithm == EncryptionAlgorithm.CHACHA20_POLY1305:
            cipher = ChaCha20Poly1305(key.key_data)
            nonce = secrets.token_bytes(12)  # 96 bits
            encrypted_bytes = cipher.encrypt(nonce, value.encode('utf-8'), None)
            
            encrypted_value = base64.b64encode(encrypted_bytes).decode('utf-8')
            iv_nonce = base64.b64encode(nonce).decode('utf-8')
            
        else:
            raise ValueError(f"Unsupported algorithm: {classification.algorithm}")
        
        return EncryptedField(
            field_name=field_name,
            encrypted_value=encrypted_value,
            key_id=key.key_id,
            algorithm=classification.algorithm,
            iv_nonce=iv_nonce,
            search_hash=search_hash
        )
    
    def decrypt_field(self, encrypted_field: EncryptedField) -> str:
        """Decrypt a field value"""
        
        if not encrypted_field.encrypted_value:
            return ""
        
        if encrypted_field.key_id == "plaintext":
            return encrypted_field.encrypted_value
        
        # Get encryption key
        if encrypted_field.key_id not in self.encryption_keys:
            raise ValueError(f"Encryption key not found: {encrypted_field.key_id}")
        
        key = self.encryption_keys[encrypted_field.key_id]
        
        try:
            # Decrypt value
            encrypted_bytes = base64.b64decode(encrypted_field.encrypted_value)
            nonce = base64.b64decode(encrypted_field.iv_nonce)
            
            if encrypted_field.algorithm == EncryptionAlgorithm.AES_256_GCM:
                cipher = AESGCM(key.key_data)
                decrypted_bytes = cipher.decrypt(nonce, encrypted_bytes, None)
                
            elif encrypted_field.algorithm == EncryptionAlgorithm.CHACHA20_POLY1305:
                cipher = ChaCha20Poly1305(key.key_data)
                decrypted_bytes = cipher.decrypt(nonce, encrypted_bytes, None)
                
            else:
                raise ValueError(f"Unsupported algorithm: {encrypted_field.algorithm}")
            
            return decrypted_bytes.decode('utf-8')
            
        except Exception as e:
            logger.error(f"Decryption failed for field {encrypted_field.field_name}: {e}")
            raise ValueError(f"Decryption failed: {e}")
    
    def mask_field(self, field_name: str, value: str) -> str:
        """Mask a field value for display"""
        
        if not value:
            return value
        
        classification = self.get_field_classification(field_name)
        if not classification or not classification.masking_pattern:
            # Default masking
            if len(value) <= 4:
                return "*" * len(value)
            else:
                return value[:2] + "*" * (len(value) - 4) + value[-2:]
        
        return classification.masking_pattern
    
    def encrypt_record(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Encrypt all classified fields in a record"""
        
        encrypted_record = record.copy()
        encryption_metadata = {}
        
        for field_name, value in record.items():
            if isinstance(value, str) and self.get_field_classification(field_name):
                encrypted_field = self.encrypt_field(field_name, value)
                
                # Store encrypted value
                encrypted_record[field_name] = encrypted_field.encrypted_value
                
                # Store encryption metadata
                encryption_metadata[field_name] = {
                    "key_id": encrypted_field.key_id,
                    "algorithm": encrypted_field.algorithm.value,
                    "iv_nonce": encrypted_field.iv_nonce,
                    "search_hash": encrypted_field.search_hash,
                    "encrypted_at": encrypted_field.encrypted_at.isoformat()
                }
        
        # Add metadata to record
        encrypted_record["_encryption_metadata"] = encryption_metadata
        
        return encrypted_record
    
    def decrypt_record(self, encrypted_record: Dict[str, Any]) -> Dict[str, Any]:
        """Decrypt all encrypted fields in a record"""
        
        decrypted_record = encrypted_record.copy()
        encryption_metadata = encrypted_record.get("_encryption_metadata", {})
        
        for field_name, metadata in encryption_metadata.items():
            if field_name in decrypted_record:
                encrypted_field = EncryptedField(
                    field_name=field_name,
                    encrypted_value=decrypted_record[field_name],
                    key_id=metadata["key_id"],
                    algorithm=EncryptionAlgorithm(metadata["algorithm"]),
                    iv_nonce=metadata["iv_nonce"],
                    search_hash=metadata.get("search_hash")
                )
                
                decrypted_record[field_name] = self.decrypt_field(encrypted_field)
        
        # Remove metadata from decrypted record
        decrypted_record.pop("_encryption_metadata", None)
        
        return decrypted_record
    
    def search_encrypted_field(self, field_name: str, search_value: str, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Search for records with matching encrypted field values"""
        
        classification = self.get_field_classification(field_name)
        if not classification or not classification.searchable:
            raise ValueError(f"Field {field_name} is not searchable")
        
        # Create search hash
        search_hash = self._create_search_hash(search_value, field_name)
        
        # Find matching records
        matching_records = []
        
        for record in records:
            metadata = record.get("_encryption_metadata", {})
            field_metadata = metadata.get(field_name, {})
            
            if field_metadata.get("search_hash") == search_hash:
                matching_records.append(record)
        
        return matching_records
    
    def rotate_keys(self, algorithm: Optional[EncryptionAlgorithm] = None):
        """Rotate encryption keys"""
        
        algorithms_to_rotate = [algorithm] if algorithm else list(EncryptionAlgorithm)
        
        for alg in algorithms_to_rotate:
            # Mark old keys as rotated
            for key in self.encryption_keys.values():
                if key.algorithm == alg and key.status == "active":
                    key.status = "rotated"
            
            # Create new key
            self._create_new_key(alg)
            
            logger.info(f"Rotated keys for algorithm: {alg.value}")
    
    def get_compliance_report(self) -> Dict[str, Any]:
        """Generate compliance report"""
        
        classification_counts = {}
        compliance_tags = set()
        
        for classification in self.field_classifications.values():
            level = classification.classification.value
            classification_counts[level] = classification_counts.get(level, 0) + 1
            compliance_tags.update(classification.compliance_tags)
        
        return {
            "total_classified_fields": len(self.field_classifications),
            "classification_breakdown": classification_counts,
            "compliance_frameworks": list(compliance_tags),
            "encryption_algorithms": [alg.value for alg in EncryptionAlgorithm],
            "active_keys": len([k for k in self.encryption_keys.values() if k.status == "active"]),
            "keys_needing_rotation": len([
                k for k in self.encryption_keys.values() 
                if k.expires_at and k.expires_at < datetime.now() + timedelta(days=7)
            ]),
            "report_generated_at": datetime.now().isoformat()
        }
    
    def audit_log_operation(self, operation: str, field_name: str, user_id: str, additional_info: Dict[str, Any] = None):
        """Log encryption/decryption operations for audit"""
        
        audit_entry = {
            "timestamp": datetime.now().isoformat(),
            "operation": operation,  # encrypt, decrypt, search, key_rotation
            "field_name": field_name,
            "user_id": user_id,
            "classification": self.get_field_classification(field_name).classification.value if self.get_field_classification(field_name) else "unclassified",
            "additional_info": additional_info or {}
        }
        
        # In production, this would go to a secure audit log system
        logger.info(f"AUDIT: {json.dumps(audit_entry)}")

# Global instance
field_encryption = None

def get_field_encryption() -> FieldEncryption:
    """Get global field encryption instance"""
    global field_encryption
    if field_encryption is None:
        field_encryption = FieldEncryption()
    return field_encryption

# Utility functions for common operations
def encrypt_pii_data(data: Dict[str, Any], user_id: str = "system") -> Dict[str, Any]:
    """Encrypt PII data in a record"""
    encryptor = get_field_encryption()
    encrypted_data = encryptor.encrypt_record(data)
    
    # Audit log
    pii_fields = [field for field in data.keys() if encryptor.get_field_classification(field)]
    for field in pii_fields:
        encryptor.audit_log_operation("encrypt", field, user_id)
    
    return encrypted_data

def decrypt_pii_data(encrypted_data: Dict[str, Any], user_id: str = "system") -> Dict[str, Any]:
    """Decrypt PII data in a record"""
    encryptor = get_field_encryption()
    decrypted_data = encryptor.decrypt_record(encrypted_data)
    
    # Audit log
    metadata = encrypted_data.get("_encryption_metadata", {})
    for field in metadata.keys():
        encryptor.audit_log_operation("decrypt", field, user_id)
    
    return decrypted_data

def mask_sensitive_data(data: Dict[str, Any]) -> Dict[str, Any]:
    """Mask sensitive data for display"""
    encryptor = get_field_encryption()
    masked_data = data.copy()
    
    for field_name, value in data.items():
        if isinstance(value, str) and encryptor.get_field_classification(field_name):
            masked_data[field_name] = encryptor.mask_field(field_name, value)
    
    return masked_data

# Example usage
if __name__ == "__main__":
    # Initialize encryption system
    encryptor = get_field_encryption()
    
    # Example customer record with PII
    customer_record = {
        "customer_id": "CUST-12345",
        "first_name": "John",
        "last_name": "Doe",
        "email": "john.doe@example.com",
        "phone": "555-123-4567",
        "ssn": "123-45-6789",
        "address": "123 Main St, Anytown, ST 12345",
        "credit_card": "4111-1111-1111-1111"
    }
    
    print("Original record:")
    print(json.dumps(customer_record, indent=2))
    
    # Encrypt PII fields
    encrypted_record = encrypt_pii_data(customer_record, "test_user")
    print("\nEncrypted record:")
    print(json.dumps({k: v for k, v in encrypted_record.items() if k != "_encryption_metadata"}, indent=2))
    
    # Decrypt record
    decrypted_record = decrypt_pii_data(encrypted_record, "test_user")
    print("\nDecrypted record:")
    print(json.dumps(decrypted_record, indent=2))
    
    # Mask sensitive fields
    masked_record = mask_sensitive_data(customer_record)
    print("\nMasked record:")
    print(json.dumps(masked_record, indent=2))
    
    # Generate compliance report
    compliance_report = encryptor.get_compliance_report()
    print("\nCompliance report:")
    print(json.dumps(compliance_report, indent=2))
    
    print("\n✅ Field-level encryption system initialized and tested")
