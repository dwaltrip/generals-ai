
# Programmatically extracting DeepNash paper

*Daniel's note: I think Claude's comment about the HTML being useless is incorrect. At least, that link looks like the full paper to me. Need to verify*

**Recipe summary:**

- **URLs:** arxiv abstract `arxiv.org/abs/2206.15378`, PDF `arxiv.org/pdf/2206.15378`. HTML at `ar5iv.labs.arxiv.org/html/2206.15378` (the standard `arxiv.org/html/...` path doesn't exist for this paper).
- **HTML is useless** — supplementary material is omitted; only high-level prose survives. PDF is the only viable path.
- **Extraction:** `curl` the PDF, then `pypdf` (no `pdftotext`/poppler on this system). Pure `PdfReader.pages[i].extract_text()` loop into a marker-tagged txt.
- **Where the spec lives:** lines ~1578–1694 of the extracted txt (Figure 7 + "Neural Implementation" subsection, pp. 38–41 of the PDF). The figure extracts messily (`stride: sskip-in` style label-gluing); the prose spec on pp. 40–41 is the cleaner read.

----
