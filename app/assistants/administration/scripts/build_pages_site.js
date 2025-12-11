#!/usr/bin/env node
/*
  Build a static site under `site/` that lists all books with HTML outputs and links to their HTML index.
  It scans both `books/` and `ragkeep-deutsche-klassik-books-de/books/` for book dirs that contain either
  `html/index.html` or `results/html/index.html`, then copies the found html dir to `site/books/<bookDir>/`.
*/

const fs = require('fs');
const path = require('path');

const REPO_ROOT = path.resolve(__dirname, '..');
const OUTPUT_DIR = path.join(REPO_ROOT, 'site');

/**
 * Parse author and title from a book directory name of the form
 * Author#Book_Title#Id (we only care about the first two segments).
 */
function parseAuthorAndTitle(dirName) {
  const parts = dirName.split('#');
  const authorRaw = parts[0] || '';
  const titleRaw = parts[1] || dirName;
  const decode = (s) => s.replace(/_/g, ' ');
  return {
    author: decode(authorRaw),
    title: decode(titleRaw),
  };
}

function ensureCleanDir(dir) {
  if (fs.existsSync(dir)) {
    fs.rmSync(dir, { recursive: true, force: true });
  }
  fs.mkdirSync(dir, { recursive: true });
}

function fileExists(p) {
  try {
    fs.accessSync(p, fs.constants.F_OK);
    return true;
  } catch {
    return false;
  }
}

function findHtmlDirForBook(bookRootDir) {
  const rootHtml = path.join(bookRootDir, 'html');
  const resultsHtml = path.join(bookRootDir, 'results', 'html');
  const rootIndex = path.join(rootHtml, 'index.html');
  const resultsIndex = path.join(resultsHtml, 'index.html');

  // Prefer html/ at book root (new default), but keep fallback to results/html (legacy)
  if (fileExists(rootIndex)) return rootHtml;
  if (fileExists(resultsIndex)) return resultsHtml;
  return null;
}

function collectBooks() {
  const sources = [
    path.join(REPO_ROOT, 'books'),
    path.join(REPO_ROOT, 'ragkeep-deutsche-klassik-books-de', 'books'),
  ];

  /** @type {Array<{dirName:string, absBookDir:string, absHtmlDir:string, relOutputDir:string, author:string, title:string}>} */
  const books = [];

  for (const source of sources) {
    if (!fs.existsSync(source)) continue;
    const entries = fs.readdirSync(source, { withFileTypes: true });
    for (const entry of entries) {
      if (!entry.isDirectory()) continue;
      const dirName = entry.name;
      const absBookDir = path.join(source, dirName);
      const absHtmlDir = findHtmlDirForBook(absBookDir);
      if (!absHtmlDir) continue;
      const { author, title } = parseAuthorAndTitle(dirName);
      const relOutputDir = path.join('books', dirName);
      books.push({ dirName, absBookDir, absHtmlDir, relOutputDir, author, title });
    }
  }

  // De-duplicate by dirName in case the same book exists in both sources; prefer the one under top-level books/ (newest pipeline output).
  const preferredPrefix = path.join(REPO_ROOT, 'books') + path.sep;
  const byName = new Map();
  for (const b of books) {
    const existing = byName.get(b.dirName);
    if (!existing) {
      byName.set(b.dirName, b);
      continue;
    }
    const isPreferred = b.absBookDir.startsWith(preferredPrefix);
    if (isPreferred) byName.set(b.dirName, b);
  }

  return Array.from(byName.values()).sort((a, b) => {
    // Sort by author then title for nicer index
    const aKey = `${a.author}\u0000${a.title}`.toLowerCase();
    const bKey = `${b.author}\u0000${b.title}`.toLowerCase();
    return aKey.localeCompare(bKey);
  });
}

function copyBookHtmlToSite(book) {
  const destAbs = path.join(OUTPUT_DIR, book.relOutputDir);
  fs.mkdirSync(path.dirname(destAbs), { recursive: true });
  fs.cpSync(book.absHtmlDir, destAbs, { recursive: true });
}

function writeNoJekyll() {
  const p = path.join(OUTPUT_DIR, '.nojekyll');
  fs.writeFileSync(p, '');
}

function writeRobots() {
  const p = path.join(OUTPUT_DIR, 'robots.txt');
  fs.writeFileSync(p, 'User-agent: *\nAllow: /\n');
}

function generateIndexHtml(books) {
  const items = books.map((b) => {
    // Encode the directory component to support special characters like '#', spaces, or umlauts
    const href = `books/${encodeURIComponent(b.dirName)}/index.html`;
    const text = `${b.author} — ${b.title}`;
    return `<li><a href="${href}">${escapeHtml(text)}</a></li>`;
  }).join('\n');

  const html = `<!doctype html>
<html lang="de">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>RAGKeep Bücher</title>
    <style>
      body { font-family: system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, "Apple Color Emoji", "Segoe UI Emoji"; margin: 2rem; line-height: 1.5; }
      header { margin-bottom: 1.5rem; }
      .muted { color: #666; font-size: 0.9rem; }
      ul { list-style: none; padding-left: 0; }
      li { margin: 0.35rem 0; }
      a { color: #0b57d0; text-decoration: none; }
      a:hover { text-decoration: underline; }
    </style>
  </head>
  <body>
    <header>
      <h1>RAGKeep – Bücher mit HTML</h1>
      <div class="muted">Automatisch generiert: ${new Date().toISOString()}</div>
    </header>
    <main>
      <ul>
        ${items}
      </ul>
    </main>
  </body>
</html>`;

  fs.writeFileSync(path.join(OUTPUT_DIR, 'index.html'), html, 'utf8');
}

function escapeHtml(s) {
  return s
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}

function main() {
  ensureCleanDir(OUTPUT_DIR);
  const books = collectBooks();
  for (const b of books) {
    copyBookHtmlToSite(b);
  }
  writeNoJekyll();
  writeRobots();
  generateIndexHtml(books);
  // eslint-disable-next-line no-console
  console.log(`Built site for ${books.length} book(s) into ${path.relative(REPO_ROOT, OUTPUT_DIR)}`);
}

main();


