import datetime as dt

import openpyxl
import pytest

from rag.config import ParsingConfig
from rag.parsing.spreadsheet import SpreadsheetParser
from rag.pipeline import IngestPipeline, QueryPipeline


def parse(path, **cfg):
    return SpreadsheetParser(ParsingConfig(**cfg)).parse(path)


def text_of(doc) -> str:
    return "\n".join(s.text for s in doc.segments)


def make_workbook(path):
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Capacity"
    ws.append(["Capacity plan 2026"])  # title row above the table
    ws.append(["Product", "Week", "Target wafers", "Start"])
    ws.append(["VRFC2501", "W1", 41.0, dt.datetime(2026, 1, 5)])
    ws.append(["VRFC2502", "W2", 42.5, dt.datetime(2026, 1, 12, 8, 30)])
    ws.append([])
    ws.append([])
    ws.append(["Owner", "Mail"])  # second table after two empty rows
    ws.append(["Planning", "plan|ning@example.com"])
    contacts = wb.create_sheet("Contacts")
    contacts.append(["Topic", "Owner"])
    contacts.append(["Scenario enquiries", "Planning desk"])
    secret = wb.create_sheet("Lookups")
    secret.append(["internal", "helper"])
    secret.append(["x", "y"])
    secret.sheet_state = "hidden"
    wb.create_sheet("Empty")
    wb.save(path)


def test_xlsx_keeps_sheet_names_tables_and_skips_hidden_and_empty(tmp_path):
    path = tmp_path / "plan.xlsx"
    make_workbook(path)
    doc = parse(path)
    body = text_of(doc)

    assert "## Capacity" in body and "## Contacts" in body
    assert "Lookups" not in body and "helper" not in body and "Empty" not in body
    assert "Title: " not in body and "Capacity plan 2026" in body
    assert "| Product | Week | Target wafers | Start |" in body
    assert "| VRFC2501 | W1 | 41 | 2026-01-05 |" in body  # whole floats lose ".0", midnight dates lose the time
    assert "| VRFC2502 | W2 | 42.5 | 2026-01-12 08:30 |" in body
    assert "| Owner | Mail |" in body  # a second table, with its own header
    assert "plan\\|ning@example.com" in body  # pipes in cells are escaped
    assert [s.page for s in doc.segments if s.text == "## Contacts"] == [2]


def test_xlsm_is_read_like_xlsx(tmp_path):
    source = tmp_path / "plan.xlsx"
    make_workbook(source)
    path = tmp_path / "plan.xlsm"
    path.write_bytes(source.read_bytes())
    assert "## Capacity" in text_of(parse(path))


def test_xls_is_read(tmp_path):
    xlwt = pytest.importorskip("xlwt")
    wb = xlwt.Workbook()
    ws = wb.add_sheet("Legacy")
    for r, row in enumerate([["Item", "Count"], ["bolts", 12], ["nuts", 7.5]]):
        for c, value in enumerate(row):
            ws.write(r, c, value)
    hidden = wb.add_sheet("Hidden")
    hidden.write(0, 0, "nope")
    hidden.visibility = 1
    path = tmp_path / "legacy.xls"
    wb.save(str(path))

    body = text_of(parse(path))
    assert "## Legacy" in body and "| bolts | 12 |" in body and "| nuts | 7.5 |" in body
    assert "nope" not in body


def test_csv_guesses_delimiter_and_encoding(tmp_path):
    path = tmp_path / "parts.csv"
    rows = 'Teil;Menge;Hinweis\nSchraube;12;"zwei\nZeilen"\nMutter;7;Größe M6\n'
    path.write_bytes(rows.encode("cp1252"))
    doc = parse(path)
    body = text_of(doc)
    assert "## " not in body  # no sheet name for csv
    assert "| Teil | Menge | Hinweis |" in body
    assert "| Schraube | 12 | zwei Zeilen |" in body
    assert "Größe M6" in body

    tsv = tmp_path / "parts.tsv"
    tsv.write_text("a\tb\n1\t2\n", encoding="utf-8")
    assert "| a | b |" in text_of(parse(tsv))


def test_row_limit_is_applied(tmp_path):
    path = tmp_path / "big.csv"
    path.write_text("n,v\n" + "\n".join(f"{i},x" for i in range(50)), encoding="utf-8")
    body = text_of(parse(path, max_sheet_rows=10))
    assert "| 8 | x |" in body and "| 20 | x |" not in body
    assert "Only the first 10 rows" in body


def test_empty_and_unreadable_inputs(tmp_path):
    empty = tmp_path / "empty.csv"
    empty.write_text("", encoding="utf-8")
    assert parse(empty).segments == []


def test_spreadsheets_are_ingested_and_cited_by_sheet(tmp_path, make_components):
    docs = tmp_path / "docs"
    docs.mkdir()
    make_workbook(docs / "plan.xlsx")
    (docs / "parts.csv").write_text("Part,Qty\nbolt,12\n", encoding="utf-8")
    c = make_components(docs)

    stats = IngestPipeline(c).run()
    assert (stats.added, stats.failed) == (2, 0)

    answer = QueryPipeline(c).ask("Who handles scenario enquiries in the Contacts sheet?")
    labels = [s.label() for s in answer.context_chunks]
    assert any(label == "plan > Contacts" for label in labels)
    assert not any("Page" in label for label in labels)
