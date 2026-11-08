"""Domain-specific exception hierarchy.

Every exception carries a ``user_message`` that is safe to show in the UI, while
the regular ``str(exc)`` may hold technical detail for logs.
"""

from __future__ import annotations


class AppError(Exception):
    """Base class for all expected, user-reportable application errors."""

    default_message = "Something went wrong."

    def __init__(self, message: str | None = None, *, user_message: str | None = None) -> None:
        super().__init__(message or self.default_message)
        self.user_message = user_message or message or self.default_message


# --- PDF -----------------------------------------------------------------------------
class PDFProcessingError(AppError):
    default_message = "The PDF could not be processed."


class InvalidPDFError(PDFProcessingError):
    default_message = "The uploaded file is not a valid PDF."


class PDFTooLargeError(PDFProcessingError):
    default_message = "The PDF exceeds the supported size."


class EncryptedPDFError(PDFProcessingError):
    default_message = "The PDF is password-protected. Please upload an unlocked copy."


class ScannedPDFError(PDFProcessingError):
    default_message = (
        "Almost no selectable text was found. The PDF looks scanned (image-only); "
        "please upload a text-based PDF or run OCR on it first."
    )


# --- LLM -----------------------------------------------------------------------------
class LLMServiceError(AppError):
    default_message = "The local language model returned an error."


class LLMConnectionError(LLMServiceError):
    default_message = "Cannot reach Ollama. Make sure it is running (`ollama serve`)."


class ModelNotFoundError(LLMServiceError):
    default_message = "The requested model is not installed in Ollama."


class LLMResponseFormatError(LLMServiceError):
    default_message = "The language model returned an unexpected response format."


# --- Storage -------------------------------------------------------------------------
class VectorStoreError(AppError):
    default_message = "The vector database (ChromaDB) reported an error."


class DatabaseError(AppError):
    default_message = "The history database reported an error."


class NotFoundError(AppError):
    default_message = "The requested item was not found."


# --- Export --------------------------------------------------------------------------
class ExportError(AppError):
    default_message = "The analysis could not be exported."
