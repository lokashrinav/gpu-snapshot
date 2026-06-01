"""
docscrawl — Discover, crawl, and aggregate documentation for any technology.

Given a topic, it:
1. Searches the web to find relevant docs, repos, blogs, papers
2. Uses an LLM to score relevance and decide which links to follow
3. Goes deep on important branches, shallow on tangential ones
4. Outputs a clean CONTEXT.md you can drop into any project

Usage:
    python docscrawl.py "cuda-checkpoint GPU snapshots gVisor" -o docs/
    python docscrawl.py "cuda-checkpoint" --seed https://docs.nvidia.com/... -o docs/
    python docscrawl.py --config crawl.json

Cost: ~$0.06 per run (Claude Haiku for relevance scoring).

Dependencies:
    pip install requests beautifulsoup4 html2text anthropic duckduckgo_search
"""

import argparse
import hashlib
import heapq
import json
import os
import re
import sys
import time
from dataclasses import dataclass, field
from urllib.parse import urljoin, urlparse, urldefrag

try:
    import requests
    from bs4 import BeautifulSoup
except ImportError:
    print("pip install requests beautifulsoup4")
    sys.exit(1)

try:
    import html2text
    _h2t = html2text.HTML2Text()
    _h2t.ignore_links = False
    _h2t.ignore_images = True
    _h2t.body_width = 0
    _h2t.ignore_emphasis = False
except ImportError:
    _h2t = None

try:
    import anthropic
except ImportError:
    anthropic = None

try:
    from duckduckgo_search import DDGS
except ImportError:
    DDGS = None


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass(order=True)
class CrawlTask:
    priority: float
    url: str = field(compare=False)
    max_depth: int = field(compare=False)
    source_type: str = field(compare=False)
    parent_url: str = field(compare=False, default="")

@dataclass
class Page:
    url: str
    content: str
    score: float
    source_type: str
    title: str = ""


# ---------------------------------------------------------------------------
# LLM client
# ---------------------------------------------------------------------------

class LLM:
    def __init__(self, model="claude-haiku-4-5-20251001", api_key=None, verbose=False):
        if anthropic is None:
            print("pip install anthropic  (needed for intelligent crawling)")
            print("Or set ANTHROPIC_API_KEY env var")
            sys.exit(1)
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        self.verbose = verbose
        self.total_input = 0
        self.total_output = 0

    def ask(self, prompt, max_tokens=2000):
        resp = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        self.total_input += resp.usage.input_tokens
        self.total_output += resp.usage.output_tokens
        text = resp.content[0].text
        if self.verbose:
            print(f"  [LLM] {self.total_input}in/{self.total_output}out tokens")
        return text

    def ask_json(self, prompt, max_tokens=2000):
        text = self.ask(prompt, max_tokens)
        match = re.search(r'\[.*\]', text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
        return None

    def cost_estimate(self):
        input_cost = self.total_input * 0.80 / 1_000_000
        output_cost = self.total_output * 4.00 / 1_000_000
        return input_cost + output_cost


# ---------------------------------------------------------------------------
# HTML → Markdown
# ---------------------------------------------------------------------------

def html_to_markdown(html_content):
    if _h2t:
        return _h2t.handle(html_content)

    soup = BeautifulSoup(html_content, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header", "aside",
                     "iframe", "noscript", "svg", "form", "button"]):
        tag.decompose()

    main = (soup.find("main") or soup.find("article") or
            soup.find(attrs={"role": "main"}) or
            soup.find("div", class_=re.compile(r"content|main|doc|body", re.I)) or
            soup.find("body") or soup)

    lines = []
    for el in main.descendants:
        if el.name in ("h1", "h2", "h3", "h4", "h5", "h6"):
            level = int(el.name[1])
            text = el.get_text(strip=True)
            if text:
                lines.append(f"\n{'#' * level} {text}\n")
        elif el.name == "p":
            text = el.get_text(strip=True)
            if text:
                lines.append(f"\n{text}\n")
        elif el.name == "li":
            text = el.get_text(strip=True)
            if text:
                lines.append(f"- {text}")
        elif el.name in ("pre", "code"):
            if el.parent and el.parent.name == "pre" and el.name == "code":
                continue
            text = el.get_text()
            if text.strip():
                lines.append(f"\n```\n{text.strip()}\n```\n")
    return "\n".join(lines)


def extract_links_with_text(html_content, base_url):
    soup = BeautifulSoup(html_content, "html.parser")
    links = []
    seen = set()
    for a in soup.find_all("a", href=True):
        href = a["href"]
        full = urljoin(base_url, href)
        full, _ = urldefrag(full)
        if full.startswith(("http://", "https://")) and full not in seen:
            seen.add(full)
            text = a.get_text(strip=True)[:100]
            links.append((full, text))
    return links


def get_page_title(html_content):
    soup = BeautifulSoup(html_content, "html.parser")
    title = soup.find("title")
    if title:
        return title.get_text(strip=True)[:200]
    h1 = soup.find("h1")
    if h1:
        return h1.get_text(strip=True)[:200]
    return ""


# ---------------------------------------------------------------------------
# Stage 1: DISCOVER — find relevant documentation sources
# ---------------------------------------------------------------------------

def generate_search_queries(topic):
    words = topic.split()
    queries = [
        topic,
        f"{topic} documentation",
        f"{topic} API reference",
        f"{topic} tutorial guide",
        f"{topic} site:github.com",
        f"{topic} example code",
    ]
    if len(words) >= 2:
        queries.append(f"{words[0]} {words[1]} blog post")
    return queries


def search_ddg(queries, max_results_per_query=10):
    if DDGS is None:
        print("  No search backend available. pip install duckduckgo_search")
        return []

    results = []
    seen = set()
    ddg = DDGS()

    for q in queries:
        try:
            hits = ddg.text(q, max_results=max_results_per_query)
            for h in hits:
                url = h.get("href", h.get("link", ""))
                if url and url not in seen:
                    seen.add(url)
                    results.append({
                        "url": url,
                        "title": h.get("title", ""),
                        "snippet": h.get("body", h.get("snippet", "")),
                    })
        except Exception as e:
            print(f"  Search failed for '{q}': {e}")
        time.sleep(1)

    return results


def search_exa(queries, max_results_per_query=10):
    try:
        from exa_py import Exa
    except ImportError:
        return []

    api_key = os.environ.get("EXA_API_KEY")
    if not api_key:
        return []

    exa = Exa(api_key)
    results = []
    seen = set()

    for q in queries:
        try:
            resp = exa.search(q, num_results=max_results_per_query, type="neural")
            for r in resp.results:
                if r.url not in seen:
                    seen.add(r.url)
                    results.append({
                        "url": r.url,
                        "title": getattr(r, "title", ""),
                        "snippet": getattr(r, "text", "")[:300],
                    })
        except Exception as e:
            print(f"  Exa search failed for '{q}': {e}")
        time.sleep(0.5)

    return results


def discover(topic, llm, seeds=None):
    print(f"\n=== DISCOVER: {topic} ===\n")

    queries = generate_search_queries(topic)
    print(f"Searching with {len(queries)} queries...")

    results = search_exa(queries)
    if not results:
        results = search_ddg(queries)

    if seeds:
        for url in seeds:
            results.append({"url": url, "title": "User-provided seed", "snippet": ""})

    if not results:
        print("No search results. Provide --seed URLs.")
        return []

    print(f"Found {len(results)} candidate URLs")

    candidates = "\n".join(
        f"{i+1}. [{r['title']}]({r['url']})\n   {r['snippet'][:150]}"
        for i, r in enumerate(results[:60])
    )

    prompt = f"""Given the research topic "{topic}", rank these URLs by relevance.

Return a JSON array of objects with fields:
- "url": the URL
- "relevance": integer 0-10
- "source_type": one of "official_docs", "github", "blog", "academic", "forum", "other"

Remove anything below relevance 4. Keep the top 30 maximum.

URLs:
{candidates}

Return ONLY the JSON array, no other text."""

    ranked = llm.ask_json(prompt, max_tokens=4000)
    if not ranked or not isinstance(ranked, list):
        print("LLM ranking failed, using all results as seeds")
        return [
            CrawlTask(priority=-5.0, url=r["url"], max_depth=1,
                       source_type="other")
            for r in results[:30]
        ]

    tasks = []
    for item in ranked:
        url = item.get("url", "")
        relevance = item.get("relevance", 5)
        source_type = item.get("source_type", "other")

        if source_type == "official_docs":
            depth = 3
        elif source_type == "github":
            depth = 2
        elif source_type == "blog":
            depth = 1
        else:
            depth = 1

        tasks.append(CrawlTask(
            priority=-relevance,  # negative because heapq is min-heap
            url=url,
            max_depth=depth,
            source_type=source_type,
        ))

    print(f"Selected {len(tasks)} seed URLs")
    for t in sorted(tasks, key=lambda t: t.priority)[:10]:
        print(f"  [{-t.priority:.0f}] {t.source_type:15s} {t.url[:80]}")

    return tasks


# ---------------------------------------------------------------------------
# Stage 2: GitHub-specific handling
# ---------------------------------------------------------------------------

GITHUB_RAW = "https://raw.githubusercontent.com"

def handle_github_repo(repo_url, session):
    parsed = urlparse(repo_url)
    parts = parsed.path.strip("/").split("/")
    if len(parts) < 2:
        return []

    owner, repo = parts[0], parts[1]
    api_base = f"https://api.github.com/repos/{owner}/{repo}"
    headers = {"Accept": "application/vnd.github.v3+json"}
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"token {token}"

    pages = []

    readme_url = f"{GITHUB_RAW}/{owner}/{repo}/HEAD/README.md"
    try:
        resp = session.get(readme_url, timeout=15)
        if resp.ok:
            pages.append(Page(
                url=f"https://github.com/{owner}/{repo}#readme",
                content=resp.text,
                score=0.9,
                source_type="github",
                title=f"{owner}/{repo} README",
            ))
    except Exception:
        pass

    for doc_dir in ["doc", "docs", "documentation"]:
        try:
            resp = session.get(f"{api_base}/contents/{doc_dir}", headers=headers, timeout=15)
            if resp.ok:
                for item in resp.json():
                    if item.get("name", "").endswith((".md", ".rst", ".txt")):
                        try:
                            fresp = session.get(item["download_url"], timeout=15)
                            if fresp.ok:
                                pages.append(Page(
                                    url=item["html_url"],
                                    content=fresp.text,
                                    score=0.8,
                                    source_type="github",
                                    title=f"{owner}/{repo}/{doc_dir}/{item['name']}",
                                ))
                        except Exception:
                            pass
        except Exception:
            pass

    for ex_dir in ["examples", "example", "demo", "demos"]:
        try:
            resp = session.get(f"{api_base}/contents/{ex_dir}", headers=headers, timeout=15)
            if resp.ok:
                for item in resp.json()[:10]:
                    name = item.get("name", "")
                    if name.endswith((".py", ".c", ".cu", ".go", ".sh", ".md")):
                        try:
                            fresp = session.get(item["download_url"], timeout=15)
                            if fresp.ok and len(fresp.text) < 50000:
                                pages.append(Page(
                                    url=item["html_url"],
                                    content=f"```\n{fresp.text}\n```",
                                    score=0.7,
                                    source_type="github",
                                    title=f"{owner}/{repo}/{ex_dir}/{name}",
                                ))
                        except Exception:
                            pass
        except Exception:
            pass

    return pages


# ---------------------------------------------------------------------------
# Stage 3: Intelligent crawl
# ---------------------------------------------------------------------------

class RateLimiter:
    def __init__(self, default_delay=1.0):
        self.last_request = {}
        self.default_delay = default_delay

    def wait(self, url):
        domain = urlparse(url).netloc
        now = time.time()
        last = self.last_request.get(domain, 0)
        delay = max(0, self.default_delay - (now - last))
        if delay > 0:
            time.sleep(delay)
        self.last_request[domain] = time.time()


def fetch(url, session, rate_limiter, timeout=30):
    rate_limiter.wait(url)
    try:
        resp = session.get(url, timeout=timeout, allow_redirects=True)
        resp.raise_for_status()
        ct = resp.headers.get("content-type", "")
        if "html" not in ct and "text" not in ct:
            return None, None
        return resp.text, resp.url
    except Exception as e:
        print(f"  SKIP {url}: {e}")
        return None, None


def keyword_score(text, topic):
    text_lower = text.lower()
    keywords = [w.lower() for w in topic.split() if len(w) > 2]
    if not keywords:
        return 0.0
    hits = sum(1 for k in keywords if k in text_lower)
    return hits / len(keywords)


def score_pages_batch(pages_to_score, topic, llm):
    if not pages_to_score:
        return {}

    entries = "\n".join(
        f"{i+1}. Title: {title}\n   Snippet: {snippet[:300]}"
        for i, (url, title, snippet) in enumerate(pages_to_score)
    )

    prompt = f"""Topic: "{topic}"

Score each page's relevance to the topic, 0-10. Return a JSON array of objects:
[{{"index": 1, "score": 8}}, ...]

Pages:
{entries}

Return ONLY the JSON array."""

    result = llm.ask_json(prompt, max_tokens=1000)
    scores = {}
    if isinstance(result, list):
        for item in result:
            idx = item.get("index", 0) - 1
            if 0 <= idx < len(pages_to_score):
                scores[pages_to_score[idx][0]] = item.get("score", 5) / 10.0
    return scores


def select_links(links_with_text, page_url, topic, llm, max_links=30):
    if not links_with_text:
        return []

    display = links_with_text[:50]
    entries = "\n".join(
        f'{i+1}. "{text}" -> {url}'
        for i, (url, text) in enumerate(display)
    )

    prompt = f"""Topic: "{topic}"
Current page: {page_url}

Which of these links are worth following to learn more about the topic?
Return a JSON array: [{{"index": 1, "priority": 8, "reason": "..."}}, ...]

Only include links with priority >= 5. Skip navigation, legal, marketing, unrelated pages.

Links:
{entries}

Return ONLY the JSON array."""

    result = llm.ask_json(prompt, max_tokens=2000)
    selected = []
    if isinstance(result, list):
        for item in result:
            idx = item.get("index", 0) - 1
            priority = item.get("priority", 5)
            if 0 <= idx < len(display) and priority >= 5:
                selected.append((display[idx][0], priority))

    return selected[:max_links]


def crawl(tasks, topic, llm, output_dir, budget=200, delay=1.0):
    print(f"\n=== CRAWL: budget={budget} pages ===\n")

    os.makedirs(os.path.join(output_dir, "pages"), exist_ok=True)

    session = requests.Session()
    session.headers["User-Agent"] = "docscrawl/2.0 (documentation aggregator)"
    rate_limiter = RateLimiter(default_delay=delay)

    heap = []
    for t in tasks:
        heapq.heappush(heap, (t.priority, id(t), t))

    visited = set()
    pages = []
    content_hashes = set()
    score_batch = []
    score_batch_interval = 10

    github_repos_done = set()

    while heap and len(pages) < budget:
        neg_pri, _, task = heapq.heappop(heap)

        url, _ = urldefrag(task.url)
        if url in visited:
            continue
        visited.add(url)

        parsed = urlparse(url)
        if parsed.netloc == "github.com" and task.source_type == "github":
            parts = parsed.path.strip("/").split("/")
            if len(parts) >= 2:
                repo_key = f"{parts[0]}/{parts[1]}"
                if repo_key not in github_repos_done:
                    github_repos_done.add(repo_key)
                    print(f"[{len(pages)+1}] GitHub repo: {repo_key}")
                    repo_pages = handle_github_repo(url, session)
                    for p in repo_pages:
                        ch = hashlib.md5(p.content[:2000].encode()).hexdigest()
                        if ch not in content_hashes:
                            content_hashes.add(ch)
                            pages.append(p)
                            print(f"  + {p.title}")
                continue

        print(f"[{len(pages)+1}/{budget}] pri={-neg_pri:.0f} {url[:90]}")

        html, final_url = fetch(url, session, rate_limiter)
        if not html:
            continue

        if final_url and final_url != url:
            visited.add(final_url)

        title = get_page_title(html)
        md = html_to_markdown(html)

        if len(md.strip()) < 100:
            print(f"  SKIP (too short)")
            continue

        if len(md) > 50000:
            md = md[:50000] + f"\n\n[Truncated — full content at {url}]"

        ch = hashlib.md5(md[:2000].encode()).hexdigest()
        if ch in content_hashes:
            print(f"  SKIP (duplicate content)")
            continue
        content_hashes.add(ch)

        ks = keyword_score(md[:5000], topic)

        if ks < 0.15:
            print(f"  LOW keyword score ({ks:.2f}), queueing for LLM scoring")
            score_batch.append((url, title, md[:500]))

            if len(score_batch) >= score_batch_interval:
                scores = score_pages_batch(score_batch, topic, llm)
                for batch_url, batch_title, batch_snippet in score_batch:
                    s = scores.get(batch_url, 0.3)
                    if s >= 0.4:
                        print(f"  LLM rescued: {batch_url[:60]} (score={s:.1f})")
                score_batch.clear()
            continue

        page_score = min(1.0, ks + 0.2)

        pages.append(Page(
            url=url,
            content=md,
            score=page_score,
            source_type=task.source_type,
            title=title,
        ))
        print(f"  KEPT (score={page_score:.2f}) {title[:60]}")

        if task.max_depth > 0 and page_score >= 0.3:
            links = extract_links_with_text(html, final_url or url)

            if len(links) > 15:
                selected = select_links(links, url, topic, llm)
            else:
                selected = [(u, 5) for u, _ in links]

            for link_url, link_priority in selected:
                if link_url not in visited:
                    child = CrawlTask(
                        priority=-link_priority,
                        url=link_url,
                        max_depth=task.max_depth - 1,
                        source_type=task.source_type,
                    )
                    heapq.heappush(heap, (child.priority, id(child), child))

    if score_batch:
        scores = score_pages_batch(score_batch, topic, llm)
        for batch_url, batch_title, _ in score_batch:
            s = scores.get(batch_url, 0.3)
            if s >= 0.4:
                print(f"  LLM late-rescue: {batch_url[:60]} (score={s:.1f})")

    print(f"\nCrawled {len(pages)} pages")
    return pages


# ---------------------------------------------------------------------------
# Stage 4: OUTPUT
# ---------------------------------------------------------------------------

def generate_summary(pages, topic, llm):
    top_pages = sorted(pages, key=lambda p: p.score, reverse=True)[:10]
    snippets = "\n\n---\n\n".join(
        f"Source: {p.url}\nTitle: {p.title}\n\n{p.content[:2000]}"
        for p in top_pages
    )

    prompt = f"""You just crawled documentation about "{topic}".

Here are the top sources found. Write a concise summary (10-15 bullet points) of:
1. What this technology is
2. Key APIs/functions and what they do
3. Important gotchas, requirements, or limitations
4. Links between the sources (e.g., "the NVIDIA API is used by gVisor's nvproxy")

Sources:
{snippets}

Write the summary as markdown bullet points. Be specific — include function names, version requirements, etc."""

    return llm.ask(prompt, max_tokens=2000)


def write_output(pages, topic, output_dir, llm):
    print(f"\n=== OUTPUT: {output_dir} ===\n")

    os.makedirs(os.path.join(output_dir, "pages"), exist_ok=True)

    pages_sorted = sorted(pages, key=lambda p: p.score, reverse=True)

    for i, p in enumerate(pages_sorted):
        slug = re.sub(r"[^a-zA-Z0-9_\-]", "_", p.title[:60] or f"page_{i}")
        filename = f"{i+1:03d}_{slug}.md"
        filepath = os.path.join(output_dir, "pages", filename)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(f"<!-- source: {p.url} -->\n")
            f.write(f"<!-- relevance: {p.score:.2f} -->\n")
            f.write(f"<!-- type: {p.source_type} -->\n\n")
            f.write(p.content)
        p._filename = filename

    print("Generating summary...")
    summary = generate_summary(pages, topic, llm)

    context_path = os.path.join(output_dir, "CONTEXT.md")
    with open(context_path, "w", encoding="utf-8") as f:
        f.write(f"# Documentation Context: {topic}\n\n")
        f.write(f"Crawled {len(pages)} pages.\n\n")

        f.write("## Summary\n\n")
        f.write(summary)
        f.write("\n\n")

        f.write("## Sources\n\n")
        f.write("| # | Score | Type | Title | URL |\n")
        f.write("|---|-------|------|-------|-----|\n")
        for i, p in enumerate(pages_sorted):
            f.write(f"| {i+1} | {p.score:.1f} | {p.source_type} | {p.title[:50]} | {p.url} |\n")
        f.write("\n---\n\n")

        for i, p in enumerate(pages_sorted):
            f.write(f"# [{i+1}] {p.title}\n")
            f.write(f"**Source**: {p.url}  \n")
            f.write(f"**Relevance**: {p.score:.1f} | **Type**: {p.source_type}\n\n")
            f.write(p.content)
            f.write("\n\n---\n\n")

    index_path = os.path.join(output_dir, "sources.json")
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump([
            {"url": p.url, "title": p.title, "score": p.score,
             "type": p.source_type, "file": getattr(p, "_filename", "")}
            for p in pages_sorted
        ], f, indent=2)

    print(f"Context: {context_path}")
    print(f"Index:   {index_path}")
    print(f"Pages:   {output_dir}/pages/")
    return context_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Discover and crawl documentation for any technology",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  python docscrawl.py "cuda-checkpoint GPU snapshots" -o docs/
  python docscrawl.py "gVisor nvproxy" --seed https://gvisor.dev/docs/user_guide/gpu/
  python docscrawl.py --config crawl.json
""")
    parser.add_argument("topic", nargs="?", help="Topic to research")
    parser.add_argument("--seed", action="append", default=[], help="Seed URL (repeatable)")
    parser.add_argument("--config", help="JSON config file")
    parser.add_argument("--output", "-o", default="docs", help="Output directory")
    parser.add_argument("--budget", "-b", type=int, default=200, help="Max pages")
    parser.add_argument("--delay", type=float, default=1.0, help="Delay between requests")
    parser.add_argument("--model", default="claude-haiku-4-5-20251001", help="LLM model for scoring")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    topic = args.topic
    seeds = args.seed
    output_dir = args.output
    budget = args.budget

    if args.config:
        with open(args.config) as f:
            cfg = json.load(f)
        topic = cfg.get("topic", topic)
        seeds = cfg.get("seeds", seeds)
        output_dir = cfg.get("output", output_dir)
        budget = cfg.get("budget", budget)
        args.delay = cfg.get("delay", args.delay)
        args.model = cfg.get("model", args.model)

    if not topic:
        parser.print_help()
        sys.exit(1)

    llm = LLM(model=args.model, verbose=args.verbose)

    tasks = discover(topic, llm, seeds=seeds)
    if not tasks:
        print("No sources found.")
        sys.exit(1)

    pages = crawl(tasks, topic, llm, output_dir, budget=budget, delay=args.delay)
    if not pages:
        print("No relevant pages found.")
        sys.exit(1)

    context_path = write_output(pages, topic, output_dir, llm)

    print(f"\nEstimated LLM cost: ${llm.cost_estimate():.3f}")
    print(f"\nDone. Drop {context_path} into your project as context.")


if __name__ == "__main__":
    main()
