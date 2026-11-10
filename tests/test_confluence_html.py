from rag.parsing.confluence_html import export_html_to_markdown, storage_to_markdown

STORAGE = """
<h1>Release process</h1>
<p>See <ac:link><ri:page ri:content-title="Build Guide" /></ac:link> first.</p>
<ac:structured-macro ac:name="toc" />
<ac:structured-macro ac:name="info">
  <ac:parameter ac:name="title">Heads up</ac:parameter>
  <ac:rich-text-body><p>Freeze is on Friday.</p></ac:rich-text-body>
</ac:structured-macro>
<ac:structured-macro ac:name="code">
  <ac:parameter ac:name="language">python</ac:parameter>
  <ac:plain-text-body><![CDATA[if a < b:
    print("x")]]></ac:plain-text-body>
</ac:structured-macro>
<table><tbody>
  <tr><th>Stage</th><th>Owner</th></tr>
  <tr><td>Build</td><td>Alice</td></tr>
</tbody></table>
<p><ac:image><ri:attachment ri:filename="flow.png" /></ac:image></p>
<ac:task-list><ac:task><ac:task-status>complete</ac:task-status><ac:task-body>Tag release</ac:task-body></ac:task></ac:task-list>
"""


def test_storage_to_markdown():
    md = storage_to_markdown(STORAGE, lambda name: "A flow diagram with 3 steps" if name == "flow.png" else None)
    assert md.startswith("# Release process")
    assert "See Build Guide first." in md
    assert "**Heads up:**" in md and "Freeze is on Friday." in md
    assert '```python\nif a < b:\n    print("x")\n```' in md
    assert "| Stage | Owner |" in md and "| Build | Alice |" in md
    assert "[Image: flow.png] A flow diagram with 3 steps" in md
    assert "[x] Tag release" in md
    assert "ac:" not in md and "ri:" not in md


EXPORT = """
<html><head><title>BSP : Capacity Planning</title></head><body>
<div id="breadcrumb-section"><ol id="breadcrumbs">
  <li><a href="index.html">BSP</a></li><li><a href="Planning_1.html">Planning</a></li>
</ol></div>
<h1 id="title-heading"><span id="title-text">BSP : Capacity Planning</span></h1>
<div id="main-content">
  <h2>Targets</h2><p>Target is 42 wafers.</p>
  <img src="attachments/123/456.png" data-linked-resource-default-alias="chart.png" />
  <img src="images/icons/bullet.gif" />
</div>
<div class="pageSection group"><h2>Attachments:</h2><a href="attachments/123/789.pdf">spec.pdf</a></div>
</body></html>
"""


def test_export_html_to_markdown():
    title, md, crumbs = export_html_to_markdown(EXPORT, lambda src: "caption" if src.endswith("456.png") else None)
    assert title == "Capacity Planning"
    assert crumbs == ["BSP", "Planning"]
    assert "## Targets" in md and "42 wafers" in md
    assert "[Image: chart.png] caption" in md
    assert "bullet" not in md and "Attachments" not in md
