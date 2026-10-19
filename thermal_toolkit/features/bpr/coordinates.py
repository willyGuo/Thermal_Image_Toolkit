"""features / bpr / coordinates."""

from pathlib import Path
import numpy as np
import os
import re


def deduplicate_coordinates(coordinates):
    unique = []
    seen = set()
    for x, y in coordinates:
        coordinate = (int(x), int(y))
        if coordinate in seen:
            continue
        seen.add(coordinate)
        unique.append(coordinate)
    return unique


def load_bpr_sie_coordinates(path):
    if not path:
        raise ValueError("Select a BPR dpc_table.bin file.")
    actual_bytes = os.path.getsize(path)
    if actual_bytes < 4 or actual_bytes % 2 != 0:
        raise ValueError(
            f"Invalid BPR BIN size: {actual_bytes:,} bytes. Expected an even-sized SIE table."
        )

    words = np.fromfile(path, dtype="<u2")
    coordinates = []
    current_y = None
    terminator_word = None
    for word_index, raw_value in enumerate(words):
        value = int(raw_value)
        if value == 0xFFFF:
            terminator_word = word_index
            break
        if value & 0x8000:
            current_y = value & 0x7FFF
            continue
        if current_y is None:
            raise ValueError(
                f"Invalid BPR SIE table: X={value} appears before a Y row marker at word {word_index}."
            )
        coordinates.append((value, current_y))

    if terminator_word is None:
        raise ValueError("Invalid BPR SIE table: 0xFFFF terminator was not found.")
    if terminator_word + 1 >= words.size or int(words[terminator_word + 1]) != 0:
        raise ValueError("Invalid BPR SIE table: expected 0x0000 after the 0xFFFF terminator.")

    unique = deduplicate_coordinates(coordinates)
    return unique, {
        "format": "Novatek SIE dpc_table.bin",
        "entry_count": len(coordinates),
        "unique_count": len(unique),
        "duplicate_count": len(coordinates) - len(unique),
        "terminator_word": terminator_word,
    }


def _parse_coordinate_integer(token):
    token = token.strip().strip("()[]{}")
    if token.lower().startswith(("0x", "+0x", "-0x")):
        return int(token, 16)
    return int(token, 10)


def load_bpr_text_coordinates(path):
    if not path:
        raise ValueError("Select a BPR coordinate CSV/TXT file.")
    coordinates = []
    with open(path, "r", encoding="utf-8-sig") as source:
        for line_number, original_line in enumerate(source, 1):
            line = original_line.split("#", 1)[0].split("//", 1)[0].strip()
            if not line:
                continue
            fields = [field for field in re.split(r"[,;\s]+", line) if field]
            if len(fields) < 2:
                raise ValueError(
                    f"Invalid coordinate at line {line_number}: expected x,y."
                )
            try:
                x = _parse_coordinate_integer(fields[0])
                y = _parse_coordinate_integer(fields[1])
            except ValueError as exc:
                if (
                    not coordinates
                    and fields[0].lower() == "x"
                    and fields[1].lower() == "y"
                ):
                    continue
                raise ValueError(
                    f"Invalid coordinate at line {line_number}: {original_line.strip()}"
                ) from exc
            coordinates.append((x, y))

    if not coordinates:
        raise ValueError("The BPR coordinate file does not contain any x,y coordinates.")
    unique = deduplicate_coordinates(coordinates)
    return unique, {
        "format": "Text coordinates (x,y)",
        "entry_count": len(coordinates),
        "unique_count": len(unique),
        "duplicate_count": len(coordinates) - len(unique),
        "terminator_word": None,
    }


def load_bpr_coordinates(path):
    suffix = Path(path).suffix.lower() if path else ""
    if suffix == ".bin":
        return load_bpr_sie_coordinates(path)
    if suffix in (".csv", ".txt"):
        return load_bpr_text_coordinates(path)
    raise ValueError("BPR coordinates must be a Novatek .bin table or an x,y .csv/.txt file.")


def split_coordinates_by_bounds(coordinates, width, height):
    valid = []
    outside = []
    for coordinate in coordinates:
        x, y = coordinate
        if 0 <= x < width and 0 <= y < height:
            valid.append(coordinate)
        else:
            outside.append(coordinate)
    return valid, outside


def format_bpr_coordinate_report(table_info, coordinates, outside_coordinates, width, height):
    outside_set = set(outside_coordinates)
    marked_count = len(coordinates) - len(outside_coordinates)
    lines = [
        f"Table entries           : {table_info['entry_count']}",
        f"Unique bad pixels       : {table_info['unique_count']}",
        f"Duplicate entries       : {table_info['duplicate_count']}",
        f"Marked in {width}x{height}      : {marked_count}",
        f"Out-of-bounds           : {len(outside_coordinates)}",
        "",
        "All unique coordinates (top-left origin; OUT = not marked):",
    ]
    coordinate_tokens = []
    index_width = max(3, len(str(len(coordinates))))
    for index, (x, y) in enumerate(coordinates, 1):
        suffix = " [OUT]" if (x, y) in outside_set else ""
        coordinate_tokens.append(
            f"{index:0{index_width}d}:({x},{y}){suffix}"
        )
    for start in range(0, len(coordinate_tokens), 4):
        lines.append("    ".join(coordinate_tokens[start:start + 4]))
    return "\n".join(lines)
