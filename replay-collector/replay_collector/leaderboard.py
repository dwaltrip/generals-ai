import json
import re

from curl_cffi import requests


RANKINGS_URL = "https://generals.io/rankings/{season_num}"


def fetch_season_state(season_num: int) -> dict:
    url = RANKINGS_URL.format(season_num=season_num)
    resp = requests.get(url, impersonate="chrome", timeout=30)
    resp.raise_for_status()
    return extract_preloaded_state(resp.text)


def extract_preloaded_state(html: str) -> dict:
    match = re.search(r"window\.__PRELOADED_STATE__\s*=\s*", html)
    if not match:
        raise ValueError("__PRELOADED_STATE__ assignment not found")

    i = match.end()
    if i >= len(html) or html[i] != "{":
        raise ValueError(f"expected '{{' at offset {i}, got {html[i:i+20]!r}")

    depth = 0
    in_string = False
    escape = False
    start = i

    while i < len(html):
        c = html[i]
        if in_string:
            if escape:
                escape = False
            elif c == "\\":
                escape = True
            elif c == '"':
                in_string = False
        else:
            if c == '"':
                in_string = True
            elif c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    return json.loads(html[start : i + 1])
        i += 1

    raise ValueError("unterminated object — reached end of input")
