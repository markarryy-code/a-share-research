"""行情资料公开能力；具体文件实现仅供组装入口使用。"""

from .domain import AdmissionSnapshot, DataError
from .ports import AdmittedReader, AdmissionArchive, MarketSource
from .application import resolve_admission, open_admitted_read

__all__ = ("AdmissionSnapshot", "DataError", "AdmittedReader", "AdmissionArchive", "MarketSource", "resolve_admission", "open_admitted_read")
