"""database_importer layout templates."""

from src.persist_layouts import (
    ALLOWED_LAYOUT_PREFIXES,
    ALLOWED_TEMPLATE_VARS,
    resolve_layout_template,
    validate_layout_template,
)


def test_database_importer_prefix_and_var():
    assert "database_importer/" in ALLOWED_LAYOUT_PREFIXES
    assert "database_id" in ALLOWED_TEMPLATE_VARS
    tmpl = "database_importer/{database_id}/{filename}"
    ok, err = validate_layout_template(tmpl)
    assert ok, err


def test_resolve_when_database():
    tmpl = resolve_layout_template(
        layout_name="default",
        layout_template=None,
        has_case=False,
        has_database=True,
    )
    assert tmpl.startswith("database_importer/")
    assert "{database_id}" in tmpl
