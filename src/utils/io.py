"""
Input/output utilities for the Assamese Instrument VLM project.

Provides small, reusable helpers for JSON, JSONL, CSV, and directory
operations used throughout dataset preparation, evaluation, and analysis.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import pandas as pd


def ensure_dir(path: str | Path) -> Path:
    """
    Create a directory if it does not already exist.

    Parameters
    ----------
    path:
        Directory path.

    Returns
    -------
    Path
        The normalized directory path.
    """
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def load_json(path: str | Path) -> Any:
    """
    Load a JSON file.

    Parameters
    ----------
    path:
        Path to the JSON file.

    Returns
    -------
    Any
        Parsed JSON content.
    """
    path = Path(path)

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def save_json(
    data: Any,
    path: str | Path,
    *,
    indent: int = 4,
    ensure_ascii: bool = False,
) -> Path:
    """
    Save data as a JSON file.

    Parent directories are created automatically.

    Parameters
    ----------
    data:
        JSON-serializable object.
    path:
        Output JSON path.
    indent:
        JSON indentation level.
    ensure_ascii:
        Whether non-ASCII characters should be escaped.

    Returns
    -------
    Path
        Path to the saved file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        json.dump(
            data,
            file,
            indent=indent,
            ensure_ascii=ensure_ascii,
        )

    return path


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """
    Load a JSON Lines (JSONL) file.

    Each non-empty line must contain one JSON object.

    Parameters
    ----------
    path:
        Path to the JSONL file.

    Returns
    -------
    list[dict[str, Any]]
        Parsed JSON objects.
    """
    path = Path(path)
    records: list[dict[str, Any]] = []

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON on line {line_number} of '{path}'."
                ) from exc

            if not isinstance(record, dict):
                raise ValueError(
                    f"Expected a JSON object on line {line_number} "
                    f"of '{path}'."
                )

            records.append(record)

    return records


def save_jsonl(
    records: Iterable[Mapping[str, Any]],
    path: str | Path,
    *,
    ensure_ascii: bool = False,
) -> Path:
    """
    Save records to a JSON Lines (JSONL) file.

    Parameters
    ----------
    records:
        Iterable of dictionary-like records.
    path:
        Output JSONL path.
    ensure_ascii:
        Whether non-ASCII characters should be escaped.

    Returns
    -------
    Path
        Path to the saved file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(
                json.dumps(
                    dict(record),
                    ensure_ascii=ensure_ascii,
                )
                + "\n"
            )

    return path


def load_csv(path: str | Path, **kwargs: Any) -> pd.DataFrame:
    """
    Load a CSV file into a pandas DataFrame.

    Additional keyword arguments are forwarded to pandas.read_csv().
    """
    return pd.read_csv(path, **kwargs)


def save_csv(
    data: pd.DataFrame,
    path: str | Path,
    *,
    index: bool = False,
    **kwargs: Any,
) -> Path:
    """
    Save a pandas DataFrame as a CSV file.

    Parent directories are created automatically.

    Additional keyword arguments are forwarded to DataFrame.to_csv().
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame.")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    data.to_csv(
        path,
        index=index,
        **kwargs,
    )

    return path