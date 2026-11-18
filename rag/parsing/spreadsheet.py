"""Spreadsheets (.xlsx .xlsm .xls .csv .tsv) as markdown tables, one section per sheet.

Only stored cell values are read. Formulas are not evaluated and macros are ignored. Hidden sheets
are skipped. Each sheet becomes a "## Sheet name" section, so the sheet name shows up in the
indexed text and in citations. Blocks of rows separated by empty rows become separate tables, and
a one-cell title row above a table is kept as a caption instead of being used as the header.
"""

from __future__ import annotations

import csv
import datetime as dt
import io
import logging
from itertools import islice
from pathlib import Path
from typing import Iterable, Iterator

from rag.config import ParsingConfig
from rag.models import ParsedDocument, Segment

log = logging.getLogger(__name__)

SPREADSHEET_SUFFIXES = {".xlsx", ".xlsm", ".xls", ".csv", ".tsv"}
_GAP = 2  # this many empty rows in a row start a new table
_MARKDOWN_STARTS = ("#", "|", "`", ">", "-", "*", "+", "=")

Rows = Iterable[list[str]]


def _cell(value) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    if isinstance(value, dt.datetime):
        return value.date().isoformat() if value.time() == dt.time(0) else value.isoformat(" ", "minutes")
    if isinstance(value, dt.date):
        return value.isoformat()
    if isinstance(value, dt.time):
        return value.strftime("%H:%M")
    if isinstance(value, float):
        if value != value:  # NaN
            return ""
        return str(int(value)) if value.is_integer() and abs(value) < 1e15 else format(value, ".10g")
    return " ".join(str(value).split())


def _xlsx_sheets(path: Path) -> Iterator[tuple[str | None, Rows]]:
    from openpyxl import load_workbook

    wb = load_workbook(path, read_only=True, data_only=True)
    try:
        for ws in wb.worksheets:
            if ws.sheet_state != "visible":
                log.debug("Skipping hidden sheet %r in %s", ws.title, path.name)
                continue
            ws.reset_dimensions()  # some writers store a wrong used range
            yield ws.title, ([_cell(v) for v in row] for row in ws.iter_rows(values_only=True))
    finally:
        wb.close()


def _xls_value(cell, datemode: int) -> str:
    import xlrd

    if cell.ctype in (xlrd.XL_CELL_EMPTY, xlrd.XL_CELL_BLANK, xlrd.XL_CELL_ERROR):
        return ""
    if cell.ctype == xlrd.XL_CELL_BOOLEAN:
        return _cell(bool(cell.value))
    if cell.ctype == xlrd.XL_CELL_DATE:
        try:
            moment = xlrd.xldate_as_datetime(cell.value, datemode)
        except Exception:
            return _cell(cell.value)
        return _cell(moment.time() if cell.value < 1 else moment)
    return _cell(cell.value)


def _xls_sheets(path: Path) -> Iterator[tuple[str | None, Rows]]:
    import xlrd

    book = xlrd.open_workbook(str(path), on_demand=True)
    try:
        for sheet in book.sheets():
            if sheet.visibility != 0:
                log.debug("Skipping hidden sheet %r in %s", sheet.name, path.name)
                continue
            yield sheet.name, ([_xls_value(c, book.datemode) for c in sheet.row(i)] for i in range(sheet.nrows))
    finally:
        book.release_resources()


def _decode(raw: bytes) -> str:
    for encoding in ("utf-8-sig", "cp1252"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw.decode("latin-1")


def _delimited_sheets(path: Path) -> Iterator[tuple[str | None, Rows]]:
    text = _decode(path.read_bytes())
    if path.suffix.lower() == ".tsv":
        delimiter = "\t"
    else:
        try:
            delimiter = csv.Sniffer().sniff(text[:8192], delimiters=",;\t|").delimiter
        except csv.Error:
            delimiter = ","
    csv.field_size_limit(10_000_000)
    reader = csv.reader(io.StringIO(text, newline=""), delimiter=delimiter)
    yield None, ([_cell(c) for c in row] for row in reader)


def _blocks(rows: list[list[str]]) -> Iterator[list[list[str]]]:
    block: list[list[str]] = []
    gap = 0
    for row in rows:
        if any(row):
            if block and gap >= _GAP:
                yield block
                block = []
            block.append(row)
            gap = 0
        else:
            gap += 1
    if block:
        yield block


def _trim_columns(block: list[list[str]]) -> list[list[str]]:
    width = max(len(r) for r in block)
    block = [r + [""] * (width - len(r)) for r in block]
    used = [i for i in range(width) if any(r[i] for r in block)]
    return [r[used[0] : used[-1] + 1] for r in block]


def _markdown_table(rows: list[list[str]]) -> str:
    def line(cells: list[str]) -> str:
        return "| " + " | ".join(c.replace("|", "\\|") for c in cells) + " |"

    return "\n".join([line(rows[0]), "|" + "---|" * len(rows[0]), *map(line, rows[1:])])


def _sheet_segments(name: str | None, number: int, rows: Rows, max_rows: int) -> list[Segment]:
    rows = list(islice(rows, max_rows + 1))
    truncated = len(rows) > max_rows
    segments: list[Segment] = []
    for block in _blocks(rows[:max_rows]):
        block = _trim_columns(block)
        if len(block) > 2 and sum(map(bool, block[0])) == 1 and sum(map(bool, block[1])) > 1:
            title = next(c for c in block[0] if c)
            segments.append(Segment(f"Title: {title}" if title.startswith(_MARKDOWN_STARTS) else title, number))
            block = block[1:]
        segments.append(Segment(_markdown_table(block), number, "table"))
    if not segments:
        return []
    if name:
        segments.insert(0, Segment(f"## {name}", number))
    if truncated:
        segments.append(Segment(f"(Only the first {max_rows} rows of this sheet were read.)", number))
        log.warning("Sheet %r has more than %d rows; the rest was not indexed", name or "", max_rows)
    return segments


class SpreadsheetParser:
    def __init__(self, cfg: ParsingConfig):
        self.cfg = cfg

    def parse(self, path: Path, title: str = "") -> ParsedDocument:
        suffix = path.suffix.lower()
        try:
            if suffix in (".xlsx", ".xlsm"):
                sheets = _xlsx_sheets(path)
            elif suffix == ".xls":
                sheets = _xls_sheets(path)
            else:
                sheets = _delimited_sheets(path)
            segments: list[Segment] = []
            for number, (name, rows) in enumerate(sheets, start=1):
                segments.extend(_sheet_segments(name, number, rows, self.cfg.max_sheet_rows))
        except ImportError as exc:
            log.warning("Skipping %s: the %r package is not installed", path.name, exc.name)
            return ParsedDocument([])
        return ParsedDocument(segments)
