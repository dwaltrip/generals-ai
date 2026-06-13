from utils.format import format_bytes, md_table


def test_format_bytes():
    assert format_bytes(0) == "0.0 B"
    assert format_bytes(1024) == "1.0 KiB"
    assert format_bytes(1024**3) == "1.0 GiB"
    # The per-batch obs figure at n=20 (bs=512, 126 ch, 32x32, fp32).
    assert format_bytes(512 * 126 * 32 * 32 * 4) == "252.0 MiB"
    # Caps at TiB rather than rolling over to a larger unit.
    assert format_bytes(1024**5) == "1024.0 TiB"


def test_md_table_emits_alignment_markers():
    # The separator row must carry GFM alignment colons so `align` actually
    # renders on GitHub (the `github` tablefmt drops them; `pipe` keeps them).
    out = md_table(["a", "b"], [["x", "1"]], align=["left", "right"])
    sep = out.splitlines()[1]
    assert sep == "|:--|--:|"

