"""Lichess puzzle dataset loader.

Downloads and caches puzzles from https://database.lichess.org/#puzzles

The puzzle database is a CSV with columns:
    PuzzleId, FEN, Moves, Rating, RatingDeviation, Popularity, NbPlays, Themes, GameUrl, OpeningTags

Example:
    puzzles = await load_lichess_puzzles(num_puzzles=100, min_rating=1500, max_rating=2000)

    for puzzle in puzzles:
        env = ChessPuzzleEnvironment(
            fen=puzzle["fen"],
            solution=puzzle["solution"],
        )
"""

import csv
from pathlib import Path
from typing import Any

import httpx

# Lichess puzzle database URL (zstandard compressed)
PUZZLE_DB_URL = "https://database.lichess.org/lichess_db_puzzle.csv.zst"

# Default cache location
DEFAULT_CACHE_DIR = Path.home() / ".cache" / "rollouts" / "lichess"


async def download_puzzles(
    cache_dir: Path = DEFAULT_CACHE_DIR,
    force: bool = False,
) -> Path:
    """Download lichess puzzle database if not cached.

    Returns path to the CSV file.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    csv_path = cache_dir / "lichess_db_puzzle.csv"

    if csv_path.exists() and not force:
        return csv_path

    # Download zstd compressed version
    zst_path = cache_dir / "lichess_db_puzzle.csv.zst"

    print(f"Downloading lichess puzzles to {zst_path}...")

    async with httpx.AsyncClient(timeout=600, follow_redirects=True) as client:
        async with client.stream("GET", PUZZLE_DB_URL) as response:
            response.raise_for_status()
            total = int(response.headers.get("content-length", 0))
            downloaded = 0

            with open(zst_path, "wb") as f:
                async for chunk in response.aiter_bytes(chunk_size=8192):
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total:
                        pct = downloaded / total * 100
                        print(f"\rDownloading: {pct:.1f}%", end="", flush=True)

    print("\nExtracting...")

    # Decompress zstd
    try:
        import zstandard as zstd

        with open(zst_path, "rb") as compressed:
            dctx = zstd.ZstdDecompressor()
            with open(csv_path, "wb") as output:
                dctx.copy_stream(compressed, output)
    except ImportError:
        # Fall back to command line zstd
        import subprocess

        result = subprocess.run(
            ["zstd", "-d", str(zst_path), "-o", str(csv_path)],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"Failed to decompress: {result.stderr}\nInstall zstandard: uv add zstandard"
            )

    # Clean up zst file
    zst_path.unlink()

    print(f"Done. Puzzles saved to {csv_path}")
    return csv_path


def parse_puzzle_row(row: dict[str, str]) -> dict[str, Any]:
    """Parse a CSV row into a puzzle dict.

    Lichess puzzle format:
        - FEN: Starting position (after opponent's move)
        - Moves: Space-separated UCI moves. First move is opponent's last move,
                 remaining moves are the solution.
    """
    moves = row["Moves"].split()

    # First move is opponent's last move - the puzzle starts AFTER this move
    # So we need to apply it to get the actual puzzle position
    opponent_move = moves[0] if moves else None
    solution_moves = moves[1:] if len(moves) > 1 else []

    # Parse themes
    themes = row.get("Themes", "").split() if row.get("Themes") else []

    return {
        "id": row["PuzzleId"],
        "fen": row["FEN"],
        "opponent_move": opponent_move,
        "solution": solution_moves,
        "rating": int(row["Rating"]) if row.get("Rating") else None,
        "rating_deviation": int(row["RatingDeviation"]) if row.get("RatingDeviation") else None,
        "popularity": int(row["Popularity"]) if row.get("Popularity") else None,
        "nb_plays": int(row["NbPlays"]) if row.get("NbPlays") else None,
        "themes": themes,
        "game_url": row.get("GameUrl", ""),
        "opening_tags": row.get("OpeningTags", "").split() if row.get("OpeningTags") else [],
    }


async def load_lichess_puzzles(
    num_puzzles: int | None = None,
    min_rating: int | None = None,
    max_rating: int | None = None,
    themes: list[str] | None = None,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    force_download: bool = False,
) -> list[dict[str, Any]]:
    """Load lichess puzzles with optional filtering.

    Args:
        num_puzzles: Maximum number of puzzles to return (None = all)
        min_rating: Minimum puzzle rating
        max_rating: Maximum puzzle rating
        themes: Filter to puzzles containing any of these themes
        cache_dir: Directory to cache downloaded puzzles
        force_download: Re-download even if cached

    Returns:
        List of puzzle dicts with keys:
            id, fen, opponent_move, solution, rating, themes, etc.
    """
    csv_path = await download_puzzles(cache_dir=cache_dir, force=force_download)

    puzzles = []

    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)

        for row in reader:
            puzzle = parse_puzzle_row(row)

            # Apply filters
            if min_rating and puzzle["rating"] and puzzle["rating"] < min_rating:
                continue
            if max_rating and puzzle["rating"] and puzzle["rating"] > max_rating:
                continue
            if themes:
                if not any(t in puzzle["themes"] for t in themes):
                    continue

            puzzles.append(puzzle)

            if num_puzzles and len(puzzles) >= num_puzzles:
                break

    return puzzles


def load_lichess_puzzles_sync(
    num_puzzles: int | None = None,
    min_rating: int | None = None,
    max_rating: int | None = None,
    themes: list[str] | None = None,
    cache_dir: Path = DEFAULT_CACHE_DIR,
) -> list[dict[str, Any]]:
    """Synchronous version of load_lichess_puzzles.

    Use this if the puzzles are already downloaded/cached.
    """
    csv_path = cache_dir / "lichess_db_puzzle.csv"

    if not csv_path.exists():
        raise FileNotFoundError(
            f"Puzzle database not found at {csv_path}. "
            "Run `await load_lichess_puzzles()` first to download."
        )

    puzzles = []

    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)

        for row in reader:
            puzzle = parse_puzzle_row(row)

            # Apply filters
            if min_rating and puzzle["rating"] and puzzle["rating"] < min_rating:
                continue
            if max_rating and puzzle["rating"] and puzzle["rating"] > max_rating:
                continue
            if themes:
                if not any(t in puzzle["themes"] for t in themes):
                    continue

            puzzles.append(puzzle)

            if num_puzzles and len(puzzles) >= num_puzzles:
                break

    return puzzles


def get_puzzle_fen_after_opponent_move(puzzle: dict[str, Any]) -> str:
    """Get the FEN after applying opponent's last move.

    Lichess puzzles start with the position BEFORE the opponent's move,
    and the puzzle is to find the best response AFTER that move.
    """
    try:
        import chess

        board = chess.Board(puzzle["fen"])
        if puzzle["opponent_move"]:
            move = chess.Move.from_uci(puzzle["opponent_move"])
            board.push(move)
        return board.fen()
    except ImportError:
        # If python-chess not available, return original FEN
        return puzzle["fen"]


# Convenience: common puzzle themes
TACTICAL_THEMES = [
    "fork",
    "pin",
    "skewer",
    "discoveredAttack",
    "doubleCheck",
    "sacrifice",
    "deflection",
    "decoy",
    "interference",
    "clearance",
    "xRayAttack",
    "zugzwang",
]

MATE_THEMES = [
    "mate",
    "mateIn1",
    "mateIn2",
    "mateIn3",
    "mateIn4",
    "mateIn5",
    "anastasiaMate",
    "arabianMate",
    "backRankMate",
    "bodenMate",
    "doubleBishopMate",
    "dovetailMate",
    "hookMate",
    "smotheredMate",
]
