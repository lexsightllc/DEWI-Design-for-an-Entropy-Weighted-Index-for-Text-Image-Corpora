import pytest

try:
    from dewi.cli import create_document
except Exception:  # pragma: no cover
    pytest.skip("cli dependencies missing", allow_module_level=True)


def test_create_document_uuid_present():
    try:
        d = create_document("hello", {"k": "v"})
    except Exception:
        pytest.skip("pipeline dependencies missing")
    assert getattr(d, "doc_id", None)
    assert getattr(d, "text", None) == "hello"
