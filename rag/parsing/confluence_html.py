"""Convert Confluence page HTML to Markdown.

Two inputs are supported:
* storage format (XHTML with ``ac:``/``ri:`` tags) returned by the REST API, and
* rendered HTML from a Confluence "HTML export" of a space.
"""

from __future__ import annotations

import re
from typing import Callable

from bs4 import BeautifulSoup, Tag
from markdownify import markdownify

ImageResolver = Callable[[str], "str | None"]

_DROP_MACROS = {
    "toc", "children", "pagetree", "recently-updated", "anchor", "attachments",
    "contentbylabel", "livesearch", "create-from-template", "gallery", "viewfile",
}
_PANEL_MACROS = {"info": "Info", "note": "Note", "warning": "Warning", "tip": "Tip", "panel": ""}


def _param(macro: Tag, name: str) -> str:
    p = macro.find("ac:parameter", attrs={"ac:name": name})
    return p.get_text(strip=True) if p else ""


def _new(soup: BeautifulSoup, name: str, text: str = "") -> Tag:
    t = soup.new_tag(name)
    if text:
        t.string = text
    return t


def _image_text(name: str, caption: str | None) -> str:
    return f"[Image: {name}] {caption}" if caption else f"[Image: {name}]"


def _to_markdown(html: str, code_blocks: list[str]) -> str:
    md = markdownify(html, heading_style="ATX", bullets="-", strip=["span"])
    for i, block in enumerate(code_blocks):
        md = md.replace(f"RAGCODEBLOCK{i}X", block)
    md = re.sub(r"[ \t]+\n", "\n", md)
    md = re.sub(r"\n{3,}", "\n\n", md)
    return md.strip()


def storage_to_markdown(xhtml: str, resolve_image: ImageResolver | None = None) -> str:
    resolve_image = resolve_image or (lambda _name: None)
    soup = BeautifulSoup(xhtml or "", "html.parser")
    code_blocks: list[str] = []

    # Innermost macros first, so nested panels/expands are handled correctly.
    for macro in reversed(soup.find_all("ac:structured-macro")):
        name = (macro.get("ac:name") or "").lower()
        body = macro.find("ac:rich-text-body")
        if name in ("code", "noformat"):
            lang = _param(macro, "language")
            plain = macro.find("ac:plain-text-body")
            code = plain.get_text() if plain else ""
            code_blocks.append(f"\n```{lang}\n{code.strip(chr(10))}\n```\n")
            macro.replace_with(_new(soup, "p", f"RAGCODEBLOCK{len(code_blocks) - 1}X"))
        elif name in _PANEL_MACROS:
            title = _param(macro, "title") or _PANEL_MACROS[name]
            quote = _new(soup, "blockquote")
            if title:
                head = _new(soup, "p")
                head.append(_new(soup, "strong", f"{title}:"))
                quote.append(head)
            if body:
                for child in list(body.children):
                    quote.append(child.extract())
            macro.replace_with(quote)
        elif name in ("expand", "ui-expand"):
            wrapper = _new(soup, "div")
            title = _param(macro, "title")
            if title:
                head = _new(soup, "p")
                head.append(_new(soup, "strong", title))
                wrapper.append(head)
            if body:
                for child in list(body.children):
                    wrapper.append(child.extract())
            macro.replace_with(wrapper)
        elif name == "jira":
            key = _param(macro, "key")
            macro.replace_with(f"[Jira {key}]" if key else "")
        elif name == "status":
            macro.replace_with(f"[{_param(macro, 'title') or 'status'}]")
        elif name in _DROP_MACROS or body is None:
            macro.decompose()
        else:
            macro.replace_with(body)
            body.unwrap()

    for img in soup.find_all("ac:image"):
        att = img.find("ri:attachment")
        url = img.find("ri:url")
        if att is not None:
            fname = att.get("ri:filename", "image")
            img.replace_with(_new(soup, "p", _image_text(fname, resolve_image(fname))))
        elif url is not None:
            img.replace_with(_new(soup, "p", f"[Image: {url.get('ri:value', '')}]"))
        else:
            img.decompose()

    for link in soup.find_all("ac:link"):
        body = link.find(["ac:link-body", "ac:plain-text-link-body"])
        target = link.find(["ri:page", "ri:attachment", "ri:user"])
        text = body.get_text(strip=True) if body else ""
        if not text and target is not None:
            text = target.get("ri:content-title") or target.get("ri:filename") or ""
        link.replace_with(text)

    for tasks in soup.find_all("ac:task-list"):
        ul = _new(soup, "ul")
        for task in tasks.find_all("ac:task"):
            done = (task.find("ac:task-status").get_text(strip=True) if task.find("ac:task-status") else "") == "complete"
            body = task.find("ac:task-body")
            ul.append(_new(soup, "li", f"[{'x' if done else ' '}] {body.get_text(' ', strip=True) if body else ''}"))
        tasks.replace_with(ul)

    for t in soup.find_all("time"):
        t.replace_with(t.get("datetime", t.get_text()))
    for emo in soup.find_all("ac:emoticon"):
        emo.decompose()
    for tag in soup.find_all(re.compile(r"^(ac|ri):")):
        tag.unwrap()

    return _to_markdown(str(soup), code_blocks)


def export_html_to_markdown(html: str, resolve_image: ImageResolver | None = None) -> tuple[str, str, list[str]]:
    """Return (title, markdown, breadcrumb) for a page from a Confluence HTML space export."""
    resolve_image = resolve_image or (lambda _src: None)
    soup = BeautifulSoup(html or "", "html.parser")

    title_el = soup.select_one("#title-text") or soup.find("title")
    title = title_el.get_text(" ", strip=True) if title_el else ""
    if " : " in title:
        title = title.split(" : ", 1)[1].strip()

    breadcrumb = [a.get_text(strip=True) for a in soup.select("#breadcrumbs li a")]

    main = soup.select_one("#main-content") or soup.body or soup
    for junk in main.select("script, style, .pageSection, .confluence-information-macro-icon"):
        junk.decompose()

    for img in main.find_all("img"):
        src = img.get("src", "")
        if not src or "/images/icons/" in src or src.startswith("images/icons"):
            img.decompose()
            continue
        name = img.get("data-linked-resource-default-alias") or src.rsplit("/", 1)[-1]
        img.replace_with(_new(soup, "p", _image_text(name, resolve_image(src))))

    code_blocks: list[str] = []
    for pre in main.find_all("pre"):
        params = pre.get("data-syntaxhighlighter-params", "")
        m = re.search(r"brush:\s*([\w+#-]+)", params)
        lang = m.group(1) if m else ""
        code_blocks.append(f"\n```{lang}\n{pre.get_text().strip(chr(10))}\n```\n")
        pre.replace_with(_new(soup, "p", f"RAGCODEBLOCK{len(code_blocks) - 1}X"))

    return title, _to_markdown(str(main), code_blocks), breadcrumb
