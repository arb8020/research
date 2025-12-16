"""Trigram index for fast code search.

Based on Russ Cox's "Regular Expression Matching with a Trigram Index"
https://swtch.com/~rsc/regexp/regexp4.html

The index maps trigrams (3-character sequences) to the files that contain them.
Query flow:
1. Extract trigrams from search query
2. Intersect posting lists to get candidate files
3. Verify candidates with actual search (tree-sitter)
"""

import sqlite3
from pathlib import Path

from csearch.backends.tree_sitter import LANG_EXTENSIONS, IGNORED_DIRS

INDEX_FILENAME = ".csearch.db"


def get_index_path(root: Path) -> Path:
    """Get path to index file for a directory."""
    return root / INDEX_FILENAME


def extract_trigrams(text: str) -> set[str]:
    """Extract all trigrams from text."""
    if len(text) < 3:
        return set()
    return {text[i : i + 3] for i in range(len(text) - 2)}


def get_index_status(root: Path) -> str:
    """Get status of index for a directory."""
    index_path = get_index_path(root)

    if not index_path.exists():
        return f"No index found at {index_path}"

    conn = sqlite3.connect(index_path)
    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(*) FROM files")
    file_count = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM trigrams")
    trigram_count = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM postings")
    posting_count = cursor.fetchone()[0]

    cursor.execute("SELECT value FROM metadata WHERE key = 'indexed_at'")
    row = cursor.fetchone()
    indexed_at = row[0] if row else "unknown"

    conn.close()

    return (
        f"Index: {index_path}\n"
        f"Files: {file_count}\n"
        f"Trigrams: {trigram_count}\n"
        f"Postings: {posting_count}\n"
        f"Indexed at: {indexed_at}"
    )


def build_index(root: Path) -> dict:
    """Build trigram index for a directory.

    Creates a SQLite database with:
    - files: (id, path, mtime) - indexed files
    - trigrams: (id, trigram) - unique trigrams
    - postings: (trigram_id, file_id) - which files contain which trigrams

    Returns stats dict with 'files' and 'trigrams' counts.
    """
    import datetime

    index_path = get_index_path(root)

    # Remove old index
    if index_path.exists():
        index_path.unlink()

    conn = sqlite3.connect(index_path)
    cursor = conn.cursor()

    # Create schema
    cursor.executescript("""
        CREATE TABLE metadata (
            key TEXT PRIMARY KEY,
            value TEXT
        );

        CREATE TABLE files (
            id INTEGER PRIMARY KEY,
            path TEXT UNIQUE NOT NULL,
            mtime REAL NOT NULL
        );

        CREATE TABLE trigrams (
            id INTEGER PRIMARY KEY,
            trigram TEXT UNIQUE NOT NULL
        );

        CREATE TABLE postings (
            trigram_id INTEGER NOT NULL,
            file_id INTEGER NOT NULL,
            PRIMARY KEY (trigram_id, file_id),
            FOREIGN KEY (trigram_id) REFERENCES trigrams(id),
            FOREIGN KEY (file_id) REFERENCES files(id)
        );

        CREATE INDEX idx_trigram_text ON trigrams(trigram);
        CREATE INDEX idx_postings_trigram ON postings(trigram_id);
    """)

    # Find all files to index
    supported_exts = set(LANG_EXTENSIONS.keys())
    files_to_index = []

    for ext in supported_exts:
        for path in root.rglob(f"*{ext}"):
            if not any(part in IGNORED_DIRS for part in path.parts):
                files_to_index.append(path)

    # Index each file
    trigram_to_id: dict[str, int] = {}
    file_count = 0

    for file_path in files_to_index:
        try:
            content = file_path.read_text(errors="ignore")
            mtime = file_path.stat().st_mtime
            rel_path = str(file_path.relative_to(root))

            # Insert file
            cursor.execute(
                "INSERT INTO files (path, mtime) VALUES (?, ?)",
                (rel_path, mtime),
            )
            file_id = cursor.lastrowid

            # Extract and insert trigrams
            trigrams = extract_trigrams(content)

            for trigram in trigrams:
                if trigram not in trigram_to_id:
                    cursor.execute(
                        "INSERT OR IGNORE INTO trigrams (trigram) VALUES (?)",
                        (trigram,),
                    )
                    cursor.execute(
                        "SELECT id FROM trigrams WHERE trigram = ?",
                        (trigram,),
                    )
                    trigram_to_id[trigram] = cursor.fetchone()[0]

                cursor.execute(
                    "INSERT OR IGNORE INTO postings (trigram_id, file_id) VALUES (?, ?)",
                    (trigram_to_id[trigram], file_id),
                )

            file_count += 1

        except Exception:
            # Skip files that can't be read
            continue

    # Store metadata
    cursor.execute(
        "INSERT INTO metadata (key, value) VALUES (?, ?)",
        ("indexed_at", datetime.datetime.now().isoformat()),
    )
    cursor.execute(
        "INSERT INTO metadata (key, value) VALUES (?, ?)",
        ("root", str(root)),
    )

    conn.commit()
    conn.close()

    return {
        "files": file_count,
        "trigrams": len(trigram_to_id),
    }


def query_index(root: Path, query: str) -> list[Path] | None:
    """Query the trigram index for candidate files.

    Args:
        root: Directory root
        query: Search query (symbol name)

    Returns:
        List of candidate file paths, or None if no index exists.
        Candidates are files that contain all trigrams in the query.
    """
    index_path = get_index_path(root)

    if not index_path.exists():
        return None

    trigrams = extract_trigrams(query)

    if not trigrams:
        # Query too short for trigrams, can't filter
        return None

    conn = sqlite3.connect(index_path)
    cursor = conn.cursor()

    # Find files containing ALL trigrams (intersection)
    # Start with files containing first trigram, then intersect with rest
    trigram_list = list(trigrams)

    # Get file IDs for first trigram
    cursor.execute(
        """
        SELECT p.file_id
        FROM postings p
        JOIN trigrams t ON p.trigram_id = t.id
        WHERE t.trigram = ?
        """,
        (trigram_list[0],),
    )
    candidate_ids = {row[0] for row in cursor.fetchall()}

    # Intersect with remaining trigrams
    for trigram in trigram_list[1:]:
        if not candidate_ids:
            break

        cursor.execute(
            """
            SELECT p.file_id
            FROM postings p
            JOIN trigrams t ON p.trigram_id = t.id
            WHERE t.trigram = ?
            """,
            (trigram,),
        )
        trigram_ids = {row[0] for row in cursor.fetchall()}
        candidate_ids &= trigram_ids

    if not candidate_ids:
        conn.close()
        return []

    # Get file paths
    placeholders = ",".join("?" * len(candidate_ids))
    cursor.execute(
        f"SELECT path FROM files WHERE id IN ({placeholders})",
        list(candidate_ids),
    )
    paths = [root / row[0] for row in cursor.fetchall()]

    conn.close()
    return paths
