"""
TrOCR 발표 대본 웹페이지(script_site/index.html) 만들기

- docs/발표대본.txt 를 슬라이드 구간별로 나눠 보여 준다.
- docs/용어정리.txt 의 용어가 대본에 나오면 옆에 [번호] 각주를 붙이고, 누르면 설명이 팝업으로 뜬다.
  (같은 용어는 대본 구간마다 처음 나올 때 한 번만 표시)
- 각 구간에서 배포된 발표 사이트의 해당 슬라이드(#sN)로 바로 갈 수 있다.
- 결과는 script_site/index.html. 이 폴더만 별도 저장소(JsonParc/ml_scrpit_two)로 올려 대본은 발표 사이트에 공개되지 않게 한다.
실행: python scripts/build_script_site.py   (텍스트를 고친 뒤 다시 실행하면 반영됨)
"""

import html
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_TXT = ROOT / "docs" / "발표대본.txt"
GLOSS_TXT = ROOT / "docs" / "용어정리.txt"
OUT = ROOT / "script_site" / "index.html"   # 별도 저장소(JsonParc/ml_scrpit_two)로 올리는 폴더
SLIDES_URL = "https://two025240014-jimin-trocr.onrender.com/"  # 배포된 발표 사이트 (Render 주소가 다르면 여기만 고친다)

# 너무 흔해서 각주를 달면 오히려 방해되는 단어
SKIP_KEYS = {"토큰"}


# ----------------------------------------------------------------- 용어 정리 읽기
def parse_glossary():
    lines = GLOSS_TXT.read_text(encoding="utf-8").splitlines()
    entries, cur, category = [], None, ""
    i = 0
    while i < len(lines):
        ln = lines[i]
        if ln.startswith("====") and i + 2 < len(lines) and lines[i + 2].startswith("===="):
            category = lines[i + 1].strip()
            i += 3
            continue
        m = re.match(r"^\[(.+)\]$", ln.strip())
        if m and i + 1 < len(lines) and lines[i + 1].startswith("----"):
            cur = {"title": m.group(1), "category": category, "body": []}
            entries.append(cur)
            i += 2
            continue
        if cur is not None and not ln.startswith("===="):
            cur["body"].append(ln)
        i += 1
    for e in entries:
        while e["body"] and not e["body"][-1].strip():
            e["body"].pop()
        e["keys"] = glossary_keys(e["title"])
    return entries


def glossary_keys(title):
    """'BPTT (Backpropagation Through Time, 시간 펼친 역전파)' -> ['BPTT', '시간 펼친 역전파']"""
    m = re.match(r"^(.*?)\s*\((.*)\)\s*$", title)
    main, paren = (m.group(1), m.group(2)) if m else (title, "")
    keys = [k.strip() for k in re.split(r"[,/]", main) if k.strip()]
    for k in [k.strip() for k in paren.split(",")]:
        if re.fullmatch(r"[A-Z][A-Za-z0-9\-]{1,7}", k) or re.search(r"[가-힣]", k):
            keys.append(k)
    out = []
    for k in keys:
        k = re.sub(r"\s*\(.*$", "", k).strip()
        if len(k) >= 2 and k not in SKIP_KEYS and k not in out:
            out.append(k)
    return out


# ----------------------------------------------------------------- 대본 읽기
def parse_script():
    lines = SCRIPT_TXT.read_text(encoding="utf-8").splitlines()
    blocks = []  # {"kind": "part"|"slide"|"text", ...}
    i = 0
    intro = []
    while i < len(lines):
        ln = lines[i]
        if ln.startswith("====") and i + 2 < len(lines) and lines[i + 2].startswith("===="):
            title = lines[i + 1].strip()
            m = re.match(r"^(.*)\((\d+)번, (.+)\)$", title)  # "표지 (1번, 30초)"
            if m:
                blocks.append({"kind": "slide", "n": int(m.group(2)), "title": m.group(1).strip(),
                               "time": m.group(3), "body": []})
            else:
                blocks.append({"kind": "part", "title": title, "body": []})
            i += 3
            continue
        m = re.match(r"^\[(\d+)번 — (.+?)(?: \((.+)\))?\]$", ln.strip())
        if m and i + 1 < len(lines) and lines[i + 1].startswith("----"):
            blocks.append({"kind": "slide", "n": int(m.group(1)), "title": m.group(2),
                           "time": m.group(3) or "", "body": []})
            i += 2
            continue
        m = re.match(r"^\[(.+)\]$", ln.strip())
        if m and i + 1 < len(lines) and lines[i + 1].startswith("----"):
            blocks.append({"kind": "sub", "title": m.group(1), "body": []})
            i += 2
            continue
        if blocks:
            blocks[-1]["body"].append(ln)
        else:
            intro.append(ln)
        i += 1
    return intro, blocks


# ----------------------------------------------------------------- 각주 붙이기
class Annotator:
    def __init__(self, entries):
        self.entries = entries
        pairs = [(k, idx) for idx, e in enumerate(entries) for k in e["keys"]]
        pairs.sort(key=lambda p: -len(p[0]))
        self.key2idx = dict(pairs)
        # 영문 용어는 단어 경계에서만 맞춘다 (TrOCR 안의 OCR, CRNN 안의 RNN에 각주가 붙지 않게)
        def pat(k):
            s = re.escape(k)
            if re.match(r"[A-Za-z0-9]", k[0]): s = r"(?<![A-Za-z0-9])" + s
            if re.match(r"[A-Za-z0-9]", k[-1]): s = s + r"(?![A-Za-z0-9\-])"
            return s
        self.rx = re.compile("|".join(pat(k) for k, _ in pairs))
        self.num = {}  # 용어 index -> 각주 번호 (페이지 전체에서 처음 붙은 순서)

    def __call__(self, text, used):
        out, pos = [], 0
        for m in self.rx.finditer(text):
            idx = self.key2idx[m.group(0)]
            if idx in used:
                continue
            used.add(idx)
            if idx not in self.num:
                self.num[idx] = len(self.num) + 1
            n = self.num[idx]
            out.append(html.escape(text[pos:m.end()]))
            out.append(f'<sup class="fn"><a href="#" data-g="{idx}">[{n}]</a></sup>')
            pos = m.end()
        out.append(html.escape(text[pos:]))
        return "".join(out)


def render_body(lines, ann):
    """(→) 는 '→ 키를 눌러 슬라이드의 다음 단계'라는 표시. 노란 칩으로 보이게 바꾼다."""
    used = set()
    base = ann
    ann = lambda t, u: base(t, u).replace("(→)", '<span class="nx" title="→ 키: 다음 단계">→</span>')
    out, para, in_list = [], [], False

    def flush():
        nonlocal para
        if para:
            out.append("<p>" + "<br>".join(ann(l, used) for l in para) + "</p>")
            para = []

    for ln in lines:
        s = ln.rstrip()
        if not s.strip():
            flush()
            if in_list:
                out.append("</ul>"); in_list = False
            continue
        lm = re.match(r"^(\s*)- (.*)$", s)
        if lm:
            flush()
            if not in_list:
                out.append("<ul>"); in_list = True
            cls = ' class="sub"' if len(lm.group(1)) >= 2 else ""
            out.append(f"<li{cls}>" + ann(lm.group(2), used) + "</li>")
            continue
        if in_list and s.startswith("  "):  # 목록 항목의 이어지는 줄 (예상 질문의 답)
            out.append('<li class="ans">' + ann(s.strip(), used) + "</li>")
            continue
        if in_list:
            out.append("</ul>"); in_list = False
        para.append(s)
    flush()
    if in_list:
        out.append("</ul>")
    return "\n".join(out)


def appendix_items(lines, ann):
    """'  - 22 부록 A 신경 활동 영상: "질문" — 설명' -> 부록 슬라이드별 항목"""
    items = []
    for ln in lines:
        m = re.match(r"^\s*- (\d+) (부록 [A-Z]) (.+?): (.*)$", ln)
        if m:
            items.append((int(m.group(1)), m.group(2), m.group(3), m.group(4)))
    out = []
    for n, tag, title, rest in items:
        out.append(f'<section class="slide-s" id="s{n}"><header><span class="badge">{n}번</span>'
                   f'<h3>{html.escape(tag)} · {html.escape(title)}</h3>'
                   f'<a class="go" href="{SLIDES_URL}#s{n}" target="slides">슬라이드 열기 ↗</a></header>'
                   f'<p>{ann(rest, set())}</p></section>')
    return "\n".join(out)


def main():
    entries = parse_glossary()
    intro, blocks = parse_script()
    ann = Annotator(entries)
    toc, body = [], []
    intro_html = render_body(intro[3:], ann)  # 제목 줄 제외
    if blocks and blocks[0]["kind"] == "part" and blocks[0]["title"] == "발표 대본":
        intro_html = render_body(blocks[0]["body"], lambda t, u: html.escape(t))  # 안내문은 각주 없이
        blocks = blocks[1:]
    for b in blocks:
        if b["kind"] == "part":
            if b["title"].startswith("부록"):
                body.append(f'<h2 class="part">{html.escape(b["title"])}</h2>')
                body.append(appendix_items(b["body"], ann))
                toc.append(f'<li class="tp">{html.escape(b["title"])}</li>')
                for ln in b["body"]:
                    m = re.match(r"^\s*- (\d+) (부록 [A-Z])", ln)
                    if m:
                        toc.append(f'<li><a href="#s{m.group(1)}">{m.group(1)} {m.group(2)}</a></li>')
                continue
            body.append(f'<h2 class="part" id="p-{len(body)}">{html.escape(b["title"])}</h2>')
            toc.append(f'<li class="tp">{html.escape(b["title"])}</li>')
            rest = render_body(b["body"], ann)
            if rest.strip():
                body.append(f'<div class="partbody">{rest}</div>')
        elif b["kind"] == "slide":
            n = b["n"]
            t = f'<span class="time">{html.escape(b["time"])}</span>' if b["time"] else ""
            body.append(f'<section class="slide-s" id="s{n}"><header><span class="badge">{n}번</span>'
                        f'<h3>{html.escape(b["title"])}</h3>{t}'
                        f'<a class="go" href="{SLIDES_URL}#s{n}" target="slides">슬라이드 열기 ↗</a></header>'
                        f'{render_body(b["body"], ann)}</section>')
            toc.append(f'<li><a href="#s{n}">{n} {html.escape(b["title"])}</a></li>')
        else:  # sub (논문 실험 상세 등)
            sid = f"x{len(body)}"
            body.append(f'<section class="slide-s plain" id="{sid}"><header><h3>{html.escape(b["title"])}</h3></header>'
                        f'{render_body(b["body"], ann)}</section>')
            toc.append(f'<li class="ts"><a href="#{sid}">{html.escape(b["title"])}</a></li>')

    nums = sorted({int(m) for m in re.findall(r'id="s(\d+)"', " ".join(body))})
    chips = "".join(f'<a class="chip" href="#s{n}" data-n="{n}">{n}</a>' for n in nums)
    gloss = [{"title": e["title"], "category": e["category"], "body": e["body"]} for e in entries]
    used = sorted(ann.num.items(), key=lambda kv: kv[1])
    fnlist = "\n".join(f'<li id="fn{n}"><a href="#" data-g="{idx}">[{n}]</a> {html.escape(entries[idx]["title"])}</li>'
                       for idx, n in used)
    page = TEMPLATE.replace("{{TOC}}", "\n".join(toc)).replace("{{BODY}}", "\n".join(body)) \
        .replace("{{INTRO}}", intro_html).replace("{{FNLIST}}", fnlist) \
        .replace("{{GLOSS}}", json.dumps(gloss, ensure_ascii=False)).replace("{{SLIDES}}", SLIDES_URL).replace("{{CHIPS}}", chips)
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(page, encoding="utf-8")
    print(f"script.html 생성: 구간 {sum(b['kind'] == 'slide' for b in blocks)}개, 각주 용어 {len(used)}개 / 용어 {len(entries)}개")


TEMPLATE = r"""<!doctype html>
<html lang="ko">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<title>TrOCR 발표 대본</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans+KR:wght@400;500;600;700&display=swap">
<style>
:root {
  --bg: #f3f5f2; --panel: #ffffff; --fg: #18222d; --muted: #5d6a75; --line: #d6ddd7;
  --accent: #d9480f; --link: #1b6f8f; --scrim: rgba(10, 14, 18, 0.45);
  --font: "IBM Plex Sans KR", "Malgun Gothic", "Apple SD Gothic Neo", sans-serif;
  --mono: "IBM Plex Mono", ui-monospace, Consolas, monospace;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) { --bg: #12171c; --panel: #1a2128; --fg: #e5eaee; --muted: #9aa6b0; --line: #2c353d; --accent: #ff7a3d; --link: #4fb3d9; --scrim: rgba(0,0,0,0.6); color-scheme: dark; }
}
:root[data-theme="dark"] { --bg: #12171c; --panel: #1a2128; --fg: #e5eaee; --muted: #9aa6b0; --line: #2c353d; --accent: #ff7a3d; --link: #4fb3d9; --scrim: rgba(0,0,0,0.6); color-scheme: dark; }
* { box-sizing: border-box; }
html { scroll-padding-top: 76px; }
body { margin: 0; background: var(--bg); color: var(--fg); font-family: var(--font); font-size: 19px; line-height: 1.8; }
body::before { content: ""; position: fixed; inset: 0 0 auto; height: env(safe-area-inset-top, 0px); z-index: 10; background: var(--panel); }
/* 패드용 상단 바: 슬라이드 번호를 눌러 바로 이동 */
.topbar { position: sticky; top: env(safe-area-inset-top, 0px); z-index: 5; background: var(--panel); border-bottom: 1px solid var(--line);
  display: flex; align-items: center; gap: 8px; padding-block: 8px;
  padding-left: max(12px, env(safe-area-inset-left, 0px)); padding-right: max(12px, env(safe-area-inset-right, 0px)); }
.chips { display: flex; gap: 6px; overflow-x: auto; flex: 1; scrollbar-width: none; -webkit-overflow-scrolling: touch; }
.chips::-webkit-scrollbar { display: none; }
.chip { flex: none; min-width: 42px; height: 42px; display: grid; place-items: center; border-radius: 8px; border: 1px solid var(--line);
  font: 600 16px var(--mono); color: var(--fg); text-decoration: none; background: var(--bg); }
.chip.on { background: var(--accent); border-color: var(--accent); color: #fff; }
.topbar button, .topbar .btn { flex: none; height: 42px; font-size: 14px; }
#wake.on { background: var(--accent); color: #fff; border-color: var(--accent); }
.wrap { display: grid; grid-template-columns: 260px minmax(0, 1fr); gap: 32px; max-width: 1180px; margin: 0 auto;
  padding-block: 28px 80px; padding-left: max(16px, env(safe-area-inset-left, 0px)); padding-right: max(16px, env(safe-area-inset-right, 0px)); }
nav.toc { position: sticky; top: 76px; align-self: start; max-height: calc(100vh - 92px); overflow-y: auto; font-size: 14px; }
nav.toc ul { list-style: none; padding: 0; margin: 0; display: grid; gap: 2px; }
nav.toc li.tp { margin-top: 12px; font-weight: 600; color: var(--accent); font-size: 13px; }
nav.toc li.ts { padding-left: 10px; }
nav.toc a { color: var(--fg); text-decoration: none; display: block; padding: 2px 6px; border-radius: 4px; }
nav.toc a:hover { background: var(--panel); }
nav.toc .tools { display: flex; gap: 6px; margin-bottom: 10px; }
button, .btn { font: 500 13px var(--font); border: 1px solid var(--line); background: var(--panel); color: var(--fg); border-radius: 6px; padding: 5px 10px; cursor: pointer; text-decoration: none; }
main { min-width: 0; }
h1 { font-size: 30px; margin: 0 0 8px; }
.intro { color: var(--muted); font-size: 15px; }
h2.part { font-size: 22px; margin: 44px 0 12px; padding-bottom: 6px; border-bottom: 2px solid var(--accent); }
.slide-s { background: var(--panel); border: 1px solid var(--line); border-radius: 10px; padding: 16px 22px; margin: 14px 0; }
.slide-s.plain { background: transparent; }
.slide-s header { display: flex; align-items: baseline; gap: 10px; flex-wrap: wrap; margin-bottom: 6px; }
.slide-s h3 { margin: 0; font-size: 19px; }
.badge { font: 500 13px var(--mono); color: #fff; background: var(--accent); border-radius: 4px; padding: 1px 7px; }
.time { font-size: 13px; color: var(--muted); }
.go { margin-left: auto; font-size: 13px; color: var(--link); text-decoration: none; }
p { margin: 8px 0; }
ul { margin: 6px 0; padding-left: 1.2em; }
li.sub { margin-left: 1.2em; list-style: circle; }
li.ans { list-style: none; color: var(--muted); margin-bottom: 6px; }
.nx { display: inline-block; min-width: 1.6em; padding: 0 6px; margin-right: 6px; border-radius: 6px; background: #ffc421; color: #2e2e2e; font-weight: 700; text-align: center; }
sup.fn { font-size: 0.68em; line-height: 0; }
sup.fn a { color: var(--link); text-decoration: none; font-weight: 600; padding: 6px 3px; font-size: 1.15em; }
sup.fn a:hover, sup.fn a:focus-visible { text-decoration: underline; }
.fnlist { columns: 2; font-size: 14px; list-style: none; padding: 0; }
.fnlist a { color: var(--link); text-decoration: none; }
:target { outline: 2px solid var(--accent); outline-offset: 2px; }
/* 각주 팝업 (위키 각주처럼) */
#pop[hidden] { display: none; }
#pop { position: fixed; inset: 0; background: var(--scrim); display: flex; align-items: center; justify-content: center; z-index: 10;
  padding-left: max(16px, env(safe-area-inset-left, 0px)); padding-right: max(16px, env(safe-area-inset-right, 0px)); }
#pop .card { background: var(--panel); color: var(--fg); border: 1px solid var(--line); border-radius: 10px; width: min(720px, 100%); max-height: 80vh; display: flex; flex-direction: column; box-shadow: 0 10px 40px rgba(0,0,0,0.3); }
#pop .inner { padding: 18px 22px; overflow-y: auto; }
#pop .cat { font-size: 12px; color: var(--accent); font-weight: 600; }
#pop h4 { margin: 2px 0 8px; font-size: 18px; }
#pop .inner p { margin: 6px 0; font-size: 17px; }
#pop .close { border: none; border-top: 1px solid var(--line); border-radius: 0 0 10px 10px; padding: 16px; font-size: 17px; width: 100%; }
@media (max-width: 1100px) { .wrap { grid-template-columns: 1fr; } nav.toc { display: none; } }
@media (max-width: 700px) { .fnlist { columns: 1; } }
</style>
</head>
<body>
<div class="topbar">
  <div class="chips" id="chips">{{CHIPS}}</div>
  <button id="wake" type="button" title="발표 중 화면이 꺼지지 않게">화면 켜짐 유지</button>
  <button id="theme" type="button">테마</button>
  <a class="btn" href="{{SLIDES}}" target="slides">슬라이드 ↗</a>
</div>
<div class="wrap">
  <nav class="toc" aria-label="목차">
    <ul>{{TOC}}<li class="tp"><a href="#fns">각주 용어 목록</a></li></ul>
  </nav>
  <main>
    <h1>TrOCR 발표 대본</h1>
    <div class="intro">{{INTRO}}<p>파란 [번호]를 누르면 용어 설명이 뜹니다. 각 구간 오른쪽 위 "슬라이드 열기"는 해당 슬라이드로 이동합니다.</p></div>
    {{BODY}}
    <h2 class="part" id="fns">각주 용어 목록</h2>
    <ul class="fnlist">{{FNLIST}}</ul>
  </main>
</div>
<div id="pop" hidden role="dialog" aria-modal="true" aria-labelledby="pop-title">
  <div class="card">
    <div class="inner"><div class="cat" id="pop-cat"></div><h4 id="pop-title"></h4><div id="pop-body"></div></div>
    <button class="close" type="button" id="pop-close">닫기</button>
  </div>
</div>
<script>
const G = {{GLOSS}};
const pop = document.getElementById("pop");
function esc(s) { return s.replace(/[&<>]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" })[c]); }
function open(i) {
  const g = G[i];
  document.getElementById("pop-cat").textContent = g.category;
  document.getElementById("pop-title").textContent = g.title;
  document.getElementById("pop-body").innerHTML = g.body.filter(l => l.trim()).map(l => "<p>" + esc(l) + "</p>").join("");
  pop.hidden = false;
  document.getElementById("pop-close").focus();
}
function close() { pop.hidden = true; }
document.addEventListener("click", e => {
  const a = e.target.closest("a[data-g]");
  if (a) { e.preventDefault(); open(+a.dataset.g); return; }
  if (e.target === pop) close();
});
document.getElementById("pop-close").onclick = close;
document.addEventListener("keydown", e => { if (e.key === "Escape") close(); });
document.getElementById("theme").onclick = () => {
  const r = document.documentElement;
  const dark = r.dataset.theme === "dark" || (!r.dataset.theme && matchMedia("(prefers-color-scheme: dark)").matches);
  r.dataset.theme = dark ? "light" : "dark";
};
// 지금 읽고 있는 구간 번호를 상단 바에서 강조
const chips = [...document.querySelectorAll(".chip")];
const secs = chips.map(c => document.getElementById("s" + c.dataset.n)).filter(Boolean);
function mark() {
  let cur = secs[0];
  for (const s of secs) if (s.getBoundingClientRect().top < 120) cur = s;
  chips.forEach(c => {
    const on = cur && c.dataset.n === cur.id.slice(1);
    // 번호 줄만 가로로 움직인다. scrollIntoView를 쓰면 페이지 스크롤이 중간에 멈춘다.
    if (on && !c.classList.contains("on")) bar.scrollTo({ left: c.offsetLeft - bar.clientWidth / 2 + c.clientWidth / 2 });
    c.classList.toggle("on", on);
  });
}
const bar = document.getElementById("chips");
// 번호를 누르면 해당 구간으로 이동
chips.forEach(c => c.addEventListener("click", e => {
  const t = document.getElementById("s" + c.dataset.n);
  if (!t) return;
  e.preventDefault();
  history.replaceState(null, "", "#s" + c.dataset.n);
  t.scrollIntoView({ block: "start" });
}));
addEventListener("scroll", mark, { passive: true });
mark();
// 화면 켜짐 유지 (Wake Lock 지원 브라우저에서만)
let lock = null;
const wake = document.getElementById("wake");
if (!("wakeLock" in navigator)) wake.hidden = true;
async function setWake(on) {
  try {
    if (on) { lock = await navigator.wakeLock.request("screen"); lock.addEventListener("release", () => { if (!document.hidden) wake.classList.remove("on"); }); }
    else if (lock) { await lock.release(); lock = null; }
    wake.classList.toggle("on", on);
  } catch (e) { wake.classList.remove("on"); }
}
wake.onclick = () => setWake(!wake.classList.contains("on"));
document.addEventListener("visibilitychange", () => { if (!document.hidden && wake.classList.contains("on")) setWake(true); });
</script>
</body>
</html>
"""

if __name__ == "__main__":
    main()
