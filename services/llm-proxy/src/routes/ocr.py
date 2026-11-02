"""
OCR Routes — اندپوینت‌های OCR برای تشخیص متن از تصویر

PaddleOCR (via Triton GPU):
    POST /ocr/detect    → تشخیص مناطق متنی (PP-OCRv5_det)
    POST /ocr/classify  → تشخیص زاویه (0°/180°)
    POST /ocr/recognize → تشخیص کاراکترها (language-specific)

Tesseract (CPU):
    POST /ocr/tesseract         → OCR کامل با Tesseract
    POST /ocr/tesseract/detect  → تشخیص زبان متن
    GET  /ocr/tesseract/langs   → لیست زبان‌های پشتیبانی شده
"""

import base64
import logging
import subprocess
import tempfile
import time
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from pydantic import BaseModel, Field

# Optional imports
try:
    import pytesseract
    from PIL import Image
    _HAS_TESSERACT = True
except ImportError:
    _HAS_TESSERACT = False

try:
    import cv2
    _HAS_CV2 = True
except ImportError:
    _HAS_CV2 = False


router = APIRouter(prefix="/ocr", tags=["OCR"])
logger = logging.getLogger(__name__)

# ============================================================================
# Pydantic Models
# ============================================================================

class OCRDetectRequest(BaseModel):
    """Request for text detection."""
    image_base64: str = Field(..., description="Base64 encoded image")
    max_side: int = Field(default=960, description="Maximum side length for resize")


class OCRDetectResponse(BaseModel):
    """Response from text detection."""
    ok: bool
    boxes: List[List[List[int]]] = Field(
        default_factory=list,
        description="List of boxes, each box is [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]"
    )
    timing_ms: float
    error: Optional[str] = None


class OCRClassifyRequest(BaseModel):
    """Request for angle classification."""
    image_base64: str = Field(..., description="Base64 encoded cropped text image")


class OCRClassifyResponse(BaseModel):
    """Response from angle classification."""
    ok: bool
    angle: int = Field(default=0, description="Detected angle: 0 or 180")
    confidence: float = Field(default=0.0)
    timing_ms: float
    error: Optional[str] = None


class OCRRecognizeRequest(BaseModel):
    """Request for text recognition."""
    image_base64: str = Field(..., description="Base64 encoded cropped text image")
    language: str = Field(default="ch", description="Language code: ch, en, arabic, cyrillic, etc.")


class OCRRecognizeResponse(BaseModel):
    """Response from text recognition."""
    ok: bool
    text: str = Field(default="")
    confidence: float = Field(default=0.0)
    timing_ms: float
    error: Optional[str] = None


class TesseractOCRRequest(BaseModel):
    """Request for Tesseract OCR."""
    image_base64: str = Field(..., description="Base64 encoded image")
    languages: List[str] = Field(
        default=["eng"],
        description="Tesseract language codes (e.g., ['eng', 'fas', 'ara'])"
    )
    config: str = Field(
        default="--psm 3",
        description="Tesseract config string"
    )


class TesseractOCRResponse(BaseModel):
    """Response from Tesseract OCR."""
    ok: bool
    text: str = Field(default="")
    languages_used: List[str] = Field(default_factory=list)
    timing_ms: float
    error: Optional[str] = None


class TesseractDetectRequest(BaseModel):
    """Request for language detection using Tesseract."""
    image_base64: str = Field(..., description="Base64 encoded image")


class TesseractDetectResponse(BaseModel):
    """Response from Tesseract language detection."""
    ok: bool
    script: str = Field(default="", description="Detected script (e.g., Latin, Arabic, Cyrillic)")
    confidence: float = Field(default=0.0)
    suggested_languages: List[str] = Field(default_factory=list)
    timing_ms: float
    error: Optional[str] = None


class LanguageListResponse(BaseModel):
    """Response listing available Tesseract languages."""
    ok: bool
    languages: List[str] = Field(default_factory=list)
    count: int = 0


# ============================================================================
# Language Mapping
# ============================================================================

# Script to PaddleOCR model mapping
SCRIPT_TO_PADDLE_MODEL = {
    "arabic": "paddleocr_v5_rec_arabic",      # عربی، فارسی، اردو
    "cyrillic": "paddleocr_v5_rec_cyrillic",  # روسی، بلغاری، ...
    "devanagari": "paddleocr_v5_rec_devanagari",  # هندی، سانسکریت
    "georgian": "paddleocr_v5_rec_ka",        # گرجی
    "japanese": "paddleocr_v5_rec_japan",     # ژاپنی
    "korean": "paddleocr_v5_rec_korean",      # کره‌ای
    "latin": "paddleocr_v5_rec_latin",        # انگلیسی، فرانسوی، ...
    "tamil": "paddleocr_v5_rec_ta",           # تامیل
    "telugu": "paddleocr_v5_rec_te",          # تلوگو
    "chinese": "paddleocr_v5_rec_ch",         # چینی
    "han": "paddleocr_v5_rec_ch",             # چینی (Han script)
}

# Tesseract script detection results to PaddleOCR model
TESS_SCRIPT_TO_PADDLE = {
    "Arabic": "arabic",
    "Cyrillic": "cyrillic",
    "Devanagari": "devanagari",
    "Georgian": "georgian",
    "Greek": "cyrillic",  # Use cyrillic model (similar)
    "Han": "chinese",
    "Hangul": "korean",
    "Hebrew": None,  # نیاز به مدل جداگانه
    "Japanese": "japanese",
    "Kannada": "devanagari",
    "Latin": "latin",
    "Tamil": "tamil",
    "Telugu": "telugu",
    "Thai": None,  # نیاز به مدل جداگانه
    "Armenian": None,  # نیاز به مدل سفارشی
}


# ============================================================================
# Helper Functions
# ============================================================================

def decode_base64_image(b64_string: str) -> np.ndarray:
    """Decode base64 image to numpy array (BGR)."""
    # Remove data URI prefix if present
    if "," in b64_string:
        b64_string = b64_string.split(",")[1]
    
    img_bytes = base64.b64decode(b64_string)
    
    if _HAS_CV2:
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return img
    else:
        # Fallback to PIL
        from PIL import Image
        pil_img = Image.open(BytesIO(img_bytes))
        return np.array(pil_img.convert("RGB"))[:, :, ::-1]


def get_tesseract_languages() -> List[str]:
    """Get list of installed Tesseract languages."""
    if not _HAS_TESSERACT:
        return []
    
    try:
        result = subprocess.run(
            ["tesseract", "--list-langs"],
            capture_output=True,
            text=True,
            timeout=10
        )
        lines = result.stdout.strip().split('\n')
        # Skip first line (header) and filter empty
        return [l.strip() for l in lines[1:] if l.strip()]
    except Exception as e:
        logger.warning(f"Failed to get Tesseract languages: {e}")
        return []


# ============================================================================
# Tesseract Endpoints (CPU)
# ============================================================================

@router.get("/tesseract/langs", response_model=LanguageListResponse)
async def list_tesseract_languages():
    """
    لیست زبان‌های پشتیبانی شده توسط Tesseract.
    شامل 124+ زبان از جمله فارسی (fas)، عربی (ara)، ارمنی (hye).
    """
    if not _HAS_TESSERACT:
        return LanguageListResponse(
            ok=False,
            languages=[],
            count=0
        )
    
    langs = get_tesseract_languages()
    return LanguageListResponse(
        ok=True,
        languages=sorted(langs),
        count=len(langs)
    )


@router.post("/tesseract", response_model=TesseractOCRResponse)
async def tesseract_ocr(request: TesseractOCRRequest):
    """
    تشخیص متن با Tesseract OCR.
    
    این سرویس روی CPU اجرا می‌شود و از 124+ زبان پشتیبانی می‌کند.
    
    Args:
        image_base64: تصویر کدشده با Base64
        languages: لیست کدهای زبان Tesseract (مثل ['eng', 'fas'])
        config: تنظیمات Tesseract
    
    Returns:
        متن استخراج شده
    """
    if not _HAS_TESSERACT:
        return TesseractOCRResponse(
            ok=False,
            text="",
            timing_ms=0,
            error="Tesseract not installed"
        )
    
    t0 = time.time()
    
    try:
        # Decode image
        img_bytes = base64.b64decode(
            request.image_base64.split(",")[1] 
            if "," in request.image_base64 
            else request.image_base64
        )
        pil_img = Image.open(BytesIO(img_bytes))
        
        # Build language string
        lang_str = "+".join(request.languages)
        
        # Run Tesseract
        text = pytesseract.image_to_string(
            pil_img,
            lang=lang_str,
            config=request.config
        )
        
        timing_ms = (time.time() - t0) * 1000
        
        return TesseractOCRResponse(
            ok=True,
            text=text.strip(),
            languages_used=request.languages,
            timing_ms=timing_ms
        )
    
    except pytesseract.TesseractError as e:
        return TesseractOCRResponse(
            ok=False,
            text="",
            timing_ms=(time.time() - t0) * 1000,
            error=f"Tesseract error: {str(e)[:200]}"
        )
    except Exception as e:
        logger.error(f"Tesseract OCR error: {e}")
        return TesseractOCRResponse(
            ok=False,
            text="",
            timing_ms=(time.time() - t0) * 1000,
            error=str(e)[:200]
        )


@router.post("/tesseract/detect", response_model=TesseractDetectResponse)
async def tesseract_detect_language(request: TesseractDetectRequest):
    """
    تشخیص زبان/اسکریپت متن در تصویر با Tesseract.
    
    از tesseract --psm 0 برای تشخیص اسکریپت استفاده می‌کند.
    
    Returns:
        script: نام اسکریپت (Latin, Arabic, Cyrillic, ...)
        suggested_languages: زبان‌های پیشنهادی برای OCR
    """
    if not _HAS_TESSERACT:
        return TesseractDetectResponse(
            ok=False,
            script="",
            timing_ms=0,
            error="Tesseract not installed"
        )
    
    t0 = time.time()
    
    try:
        # Decode image
        img_bytes = base64.b64decode(
            request.image_base64.split(",")[1] 
            if "," in request.image_base64 
            else request.image_base64
        )
        pil_img = Image.open(BytesIO(img_bytes))
        
        # Use OSD (Orientation and Script Detection)
        try:
            osd = pytesseract.image_to_osd(pil_img, output_type=pytesseract.Output.DICT)
            script = osd.get('script', 'Unknown')
            confidence = osd.get('script_conf', 0.0)
        except pytesseract.TesseractError:
            # Fallback: try to detect via sampling different langs
            script = "Unknown"
            confidence = 0.0
        
        # Map script to suggested languages
        suggested = []
        if script == "Arabic":
            suggested = ["ara", "fas", "urd"]
        elif script == "Cyrillic":
            suggested = ["rus", "ukr", "bul"]
        elif script == "Latin":
            suggested = ["eng", "fra", "deu", "spa"]
        elif script == "Han":
            suggested = ["chi_sim", "chi_tra"]
        elif script == "Japanese":
            suggested = ["jpn"]
        elif script == "Hangul":
            suggested = ["kor"]
        elif script == "Devanagari":
            suggested = ["hin", "san"]
        elif script == "Armenian":
            suggested = ["hye"]
        elif script == "Georgian":
            suggested = ["kat"]
        elif script == "Hebrew":
            suggested = ["heb"]
        elif script == "Tamil":
            suggested = ["tam"]
        elif script == "Telugu":
            suggested = ["tel"]
        else:
            suggested = ["eng"]  # Default
        
        timing_ms = (time.time() - t0) * 1000
        
        return TesseractDetectResponse(
            ok=True,
            script=script,
            confidence=float(confidence),
            suggested_languages=suggested,
            timing_ms=timing_ms
        )
    
    except Exception as e:
        logger.error(f"Language detection error: {e}")
        return TesseractDetectResponse(
            ok=False,
            script="",
            timing_ms=(time.time() - t0) * 1000,
            error=str(e)[:200]
        )


# ============================================================================
# PaddleOCR Endpoints (via Triton GPU) — Placeholder
# These will call the /v1/infer endpoint for actual inference
# ============================================================================

@router.post("/detect", response_model=OCRDetectResponse)
async def paddle_detect(request: OCRDetectRequest):
    """
    تشخیص مناطق متنی با PP-OCRv5_det.
    
    این endpoint فقط مناطق متنی را پیدا می‌کند.
    برای inference واقعی از /v1/infer استفاده کنید.
    
    TODO: Implement direct Triton call
    """
    return OCRDetectResponse(
        ok=False,
        boxes=[],
        timing_ms=0,
        error="Use /v1/infer with model='paddleocr_v5_det'"
    )


@router.post("/classify", response_model=OCRClassifyResponse)
async def paddle_classify(request: OCRClassifyRequest):
    """
    تشخیص زاویه متن (0° یا 180°) با PP-OCRv5_cls.
    
    TODO: Implement direct Triton call
    """
    return OCRClassifyResponse(
        ok=False,
        angle=0,
        confidence=0.0,
        timing_ms=0,
        error="Use /v1/infer with model='paddleocr_v5_cls'"
    )


@router.post("/recognize", response_model=OCRRecognizeResponse)
async def paddle_recognize(request: OCRRecognizeRequest):
    """
    تشخیص کاراکترها با مدل PP-OCRv5 مخصوص زبان.
    
    Language codes:
        - ch: Chinese
        - en: English  
        - arabic: Arabic/Persian/Urdu
        - cyrillic: Russian/Ukrainian/Bulgarian
        - devanagari: Hindi/Sanskrit
        - ka: Georgian
        - japan: Japanese
        - korean: Korean
        - latin: Latin script languages
        - ta: Tamil
        - te: Telugu
    
    TODO: Implement direct Triton call
    """
    model_name = f"paddleocr_v5_rec_{request.language}"
    
    return OCRRecognizeResponse(
        ok=False,
        text="",
        confidence=0.0,
        timing_ms=0,
        error=f"Use /v1/infer with model='{model_name}'"
    )


# ============================================================================
# Full Pipeline Endpoint
# ============================================================================

class FullOCRRequest(BaseModel):
    """Request for full OCR pipeline."""
    image_base64: str = Field(..., description="Base64 encoded image")
    auto_detect_language: bool = Field(
        default=True,
        description="Use Tesseract to auto-detect language"
    )
    languages: Optional[List[str]] = Field(
        default=None,
        description="Override languages (Tesseract codes)"
    )


class OCRRegion(BaseModel):
    """Single OCR result region."""
    text: str
    confidence: float
    box: List[List[int]]
    language: str


class FullOCRResponse(BaseModel):
    """Response from full OCR pipeline."""
    ok: bool
    regions: List[OCRRegion] = Field(default_factory=list)
    full_text: str = Field(default="")
    detected_script: Optional[str] = None
    timing_ms: Dict[str, float] = Field(default_factory=dict)
    error: Optional[str] = None


@router.post("/full", response_model=FullOCRResponse)
async def full_ocr_pipeline(request: FullOCRRequest):
    """
    پایپلاین کامل OCR:
    
    1. Detection: تشخیص مناطق متنی (PP-OCRv5_det via Triton)
    2. Language Detection: تشخیص زبان (Tesseract CPU)
    3. Recognition: تشخیص کاراکترها (PP-OCRv5_rec_* via Triton)
    
    اگر auto_detect_language=true باشد، زبان به طور خودکار تشخیص داده می‌شود.
    """
    # TODO: Implement full pipeline
    # For now, use Tesseract as fallback
    
    if not _HAS_TESSERACT:
        return FullOCRResponse(
            ok=False,
            error="Full pipeline not implemented yet. Use individual endpoints."
        )
    
    t_start = time.time()
    timings = {}
    
    try:
        # Decode image
        t0 = time.time()
        img_bytes = base64.b64decode(
            request.image_base64.split(",")[1] 
            if "," in request.image_base64 
            else request.image_base64
        )
        pil_img = Image.open(BytesIO(img_bytes))
        timings["decode"] = (time.time() - t0) * 1000
        
        # Language detection
        t0 = time.time()
        script = "Latin"
        if request.auto_detect_language and request.languages is None:
            try:
                osd = pytesseract.image_to_osd(pil_img, output_type=pytesseract.Output.DICT)
                script = osd.get('script', 'Latin')
            except:
                script = "Latin"
        timings["lang_detect"] = (time.time() - t0) * 1000
        
        # Determine languages
        if request.languages:
            langs = request.languages
        elif script == "Arabic":
            langs = ["ara", "fas"]
        elif script == "Cyrillic":
            langs = ["rus"]
        elif script == "Han":
            langs = ["chi_sim"]
        elif script == "Japanese":
            langs = ["jpn"]
        elif script == "Armenian":
            langs = ["hye"]
        else:
            langs = ["eng"]
        
        # OCR with Tesseract (as fallback)
        t0 = time.time()
        lang_str = "+".join(langs)
        text = pytesseract.image_to_string(pil_img, lang=lang_str)
        timings["recognition"] = (time.time() - t0) * 1000
        
        timings["total"] = (time.time() - t_start) * 1000
        
        return FullOCRResponse(
            ok=True,
            regions=[OCRRegion(
                text=text.strip(),
                confidence=0.9,  # Tesseract doesn't provide per-region confidence easily
                box=[[0, 0], [pil_img.width, 0], [pil_img.width, pil_img.height], [0, pil_img.height]],
                language=langs[0]
            )],
            full_text=text.strip(),
            detected_script=script,
            timing_ms=timings
        )
    
    except Exception as e:
        logger.error(f"Full OCR pipeline error: {e}")
        return FullOCRResponse(
            ok=False,
            error=str(e)[:200],
            timing_ms={"total": (time.time() - t_start) * 1000}
        )
