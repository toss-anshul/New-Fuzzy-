from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

try:
    from dotenv import load_dotenv
except ModuleNotFoundError:  # safety if dotenv not installed
    def load_dotenv(*args, **kwargs):
        return False

import ocr_service

extract_ocr_from_path = ocr_service.extract_ocr_from_path
DEFAULT_MAX_WORKERS = ocr_service.DEFAULT_MAX_WORKERS
DEFAULT_PDF_DPI = ocr_service.DEFAULT_PDF_DPI
DEFAULT_BATCH_SIZE = ocr_service.DEFAULT_BATCH_SIZE


# Load environment variables from a .env file if present
load_dotenv()

app = FastAPI(
    title="Gemini OCR API",
    description="Simple HTTP API wrapper around the Gemini OCR PoC, with keyword matching and simple field verification.",
    version="1.1.0",
)

# Allow cross-origin requests from the frontend (development convenience)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/ocr/keywords")
async def ocr_with_keywords(
    file: UploadFile = File(..., description="PDF or image file"),
    keywords: Optional[str] = Form(
        default="",
        description="Comma-separated list of words/phrases to search for in the document.",
    ),
    model: str = Form("gemini-2.5-flash-lite"),
    chunk_size: int = Form(1200),
) -> JSONResponse:
    """
    Run OCR on the uploaded file and optionally search for given keywords
    across the entire extracted text.
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=500,
            detail="GEMINI_API_KEY is not set on the server.",
        )

    # Save upload to a temporary file
    suffix = Path(file.filename or "upload.bin").suffix or ".bin"
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(await file.read())
            tmp_path = Path(tmp.name)

        ocr_result = extract_ocr_from_path(
            input_path=tmp_path,
            api_key=api_key,
            model=model,
            chunk_size=int(chunk_size),
            max_workers=int(DEFAULT_MAX_WORKERS),
            pdf_dpi=int(DEFAULT_PDF_DPI),
            quality_boost=True,
            batch_size=int(DEFAULT_BATCH_SIZE),
            use_cache=True,
            use_async=True,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        try:
            tmp_path.unlink(missing_ok=True)  # type: ignore[name-defined]
        except Exception:
            pass

    payload = ocr_result.model_dump()

    # Keyword / phrase search
    raw_keywords = keywords or ""
    keyword_list: List[str] = [
        k.strip() for k in raw_keywords.split(",") if k.strip()
    ]

    keyword_results = []
    if keyword_list:
        pages_lower = [
            (p.page_number, (p.text or "").lower())
            for p in ocr_result.pages
        ]
        full_text_lower = (ocr_result.full_text or "").lower()

        for kw in keyword_list:
            kw_lower = kw.lower()
            pages_found = [
                num for num, text in pages_lower
                if kw_lower in text
            ]
            found = bool(pages_found) or (kw_lower in full_text_lower)
            keyword_results.append(
                {
                    "keyword": kw,
                    "found": found,
                    "pages": pages_found,
                }
            )

    response_body = {
        "ocr": payload,
        "keywords": {
            "input": keyword_list,
            "results": keyword_results,
        },
    }

    return JSONResponse(response_body)


def _normalize_spaces(text: str) -> str:
    """Remove whitespace and lowercase for robust matching."""
    return "".join(ch for ch in text.lower() if not ch.isspace())


@app.post("/ocr/bank-verify")
async def ocr_bank_verify(
    file: UploadFile = File(..., description="Bank proof document (PDF or image)"),
    account_number: str = Form(...),
    ifsc_code: str = Form(...),
    bank_holder_name: Optional[str] = Form(None),
    bank_name: Optional[str] = Form(None),
    branch: Optional[str] = Form(None),
    model: str = Form("gemini-2.5-flash-lite"),
    chunk_size: int = Form(1200),
) -> JSONResponse:
    """
    Run OCR on the uploaded bank document and verify that the extracted text
    contains the provided account details.

    A field is considered matched if its value (after basic normalisation) is
    found anywhere in the OCR'd text.
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=500,
            detail="GEMINI_API_KEY is not set on the server.",
        )

    suffix = Path(file.filename or "upload.bin").suffix or ".bin"
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(await file.read())
            tmp_path = Path(tmp.name)

        ocr_result = extract_ocr_from_path(
            input_path=tmp_path,
            api_key=api_key,
            model=model,
            chunk_size=int(chunk_size),
            max_workers=int(DEFAULT_MAX_WORKERS),
            pdf_dpi=int(DEFAULT_PDF_DPI),
            quality_boost=True,
            batch_size=int(DEFAULT_BATCH_SIZE),
            use_cache=True,
            use_async=True,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        try:
            tmp_path.unlink(missing_ok=True)  # type: ignore[name-defined]
        except Exception:
            pass

    full_text = ocr_result.full_text or ""
    full_text_norm = _normalize_spaces(full_text)

    def match_account(num: str) -> bool:
        value_norm = _normalize_spaces(num)
        return bool(value_norm) and value_norm in full_text_norm

    def match_simple(value: Optional[str]) -> bool:
        if not value:
            return False
        return value.strip().lower() in full_text.lower()

    # Individual field matches
    account_match = match_account(account_number)
    ifsc_match = match_simple(ifsc_code)
    holder_match = match_simple(bank_holder_name)
    bank_name_match = match_simple(bank_name)
    branch_match = match_simple(branch)

    field_matches = {
        "accountNumber": {
            "value": account_number,
            "matched": account_match,
        },
        "ifscCode": {
            "value": ifsc_code,
            "matched": ifsc_match,
        },
        "bankHolderName": {
            "value": bank_holder_name,
            "matched": holder_match if bank_holder_name else None,
        },
        "bankName": {
            "value": bank_name,
            "matched": bank_name_match if bank_name else None,
        },
        "bankBranch": {
            "value": branch,
            "matched": branch_match if branch else None,
        },
    }

    # Overall verification: all required fields (account + IFSC) must match
    # Optional fields, if provided, also need to match.
    required_ok = account_match and ifsc_match
    optional_ok = True
    if bank_holder_name:
        optional_ok = optional_ok and holder_match
    if bank_name:
        optional_ok = optional_ok and bank_name_match
    if branch:
        optional_ok = optional_ok and branch_match

    verified = required_ok and optional_ok

    response_body = {
        "verified": verified,
        "fields": field_matches,
        "ocrPreviewChars": len(full_text),
    }

    return JSONResponse(response_body)


@app.post("/ocr/credit-card-verify")
async def ocr_credit_card_verify(
    file: UploadFile = File(..., description="Credit card statement (PDF or image)"),
    card_number: str = Form(...),
    holder_name: Optional[str] = Form(None),
    bank_name: Optional[str] = Form(None),
    branch: Optional[str] = Form(None),
    model: str = Form("gemini-2.5-flash-lite"),
    chunk_size: int = Form(1200),
) -> JSONResponse:
    """
    Run OCR on the uploaded credit card statement and verify that the extracted text
    contains the provided card details (card number, bank name, optional branch/name).
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=500,
            detail="GEMINI_API_KEY is not set on the server.",
        )

    suffix = Path(file.filename or "upload.bin").suffix or ".bin"
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(await file.read())
            tmp_path = Path(tmp.name)

        ocr_result = extract_ocr_from_path(
            input_path=tmp_path,
            api_key=api_key,
            model=model,
            chunk_size=int(chunk_size),
            max_workers=int(DEFAULT_MAX_WORKERS),
            pdf_dpi=int(DEFAULT_PDF_DPI),
            quality_boost=True,
            batch_size=int(DEFAULT_BATCH_SIZE),
            use_cache=True,
            use_async=True,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        try:
            tmp_path.unlink(missing_ok=True)  # type: ignore[name-defined]
        except Exception:
            pass

    full_text = ocr_result.full_text or ""
    full_text_norm = _normalize_spaces(full_text)

    def match_card(num: str) -> bool:
        value_norm = _normalize_spaces(num)
        return bool(value_norm) and value_norm in full_text_norm

    def match_simple(value: Optional[str]) -> bool:
        if not value:
            return False
        return value.strip().lower() in full_text.lower()

    card_match = match_card(card_number)
    bank_match = match_simple(bank_name)
    branch_match = match_simple(branch)
    holder_match = match_simple(holder_name)

    field_matches = {
        "creditCardNumber": {"value": card_number, "matched": card_match},
        "bankName": {
            "value": bank_name,
            "matched": bank_match if bank_name else None,
        },
        "bankBranch": {
            "value": branch,
            "matched": branch_match if branch else None,
        },
        "cardHolderName": {
            "value": holder_name,
            "matched": holder_match if holder_name else None,
        },
    }

    # Required: card number must match.
    # Holder name, bank name, and branch are optional signals – if provided, we
    # check them and surface match flags, but they do NOT block overall success.
    required_ok = card_match
    verified = required_ok

    response_body = {
        "verified": verified,
        "fields": field_matches,
        "ocrPreviewChars": len(full_text),
    }

    return JSONResponse(response_body)


