from __future__ import annotations
from pathlib import Path
import os, sys, json

def choose_input_file(input_dir: str,
                      allowed_ext: tuple[str, ...] = ('.tek', '.txt', '.csv')) -> str | None:
    
    env_path = os.getenv("INPUT_FILE")
    if env_path:
        p = Path(env_path)
        if not p.is_absolute():
            p = Path(input_dir) / p
        if p.is_file():
            return str(p.resolve())
        else:
            print(f"INPUT_FILE ei löydy: {p}")

    cfg_path = Path("aineiston_kasittely/config_files/config.json")
    if cfg_path.exists():
        try:
            cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
            cand = cfg.get("input_file")
            if cand:
                p = Path(cand)
                if not p.is_absolute():
                    p = Path(input_dir) / p
                if p.is_file():
                    return str(p.resolve())
                else:
                    print(f"config.json input_file ei löydy: {p}")
        except Exception as e:
            print(f"config.json lukeminen epäonnistui: {e}")

    input_dir = Path(input_dir)
    if not input_dir.exists():
        print(f"Input-hakemistoa ei ole: {input_dir}")
        return None

    files = [p for p in input_dir.iterdir()
             if p.is_file() and p.suffix.lower() in allowed_ext]
    if not files:
        print(f"Ei löydy tiedostoja, joilla päätteet {allowed_ext} hakemistossa {input_dir}")
        return None

    files.sort(key=lambda p: p.stat().st_mtime, reverse=True)

    if not sys.stdin.isatty():
        chosen = files[0]
        print(f"Auto-selected latest file: {chosen.name}")
        return str(chosen.resolve())

    print("Available input files:")
    for i, p in enumerate(files, 1):
        print(f"  {i}) {p.name}")
    try:
        s = input("Choose the file number (press Enter for 1): ").strip()
        idx = 1 if s == "" else int(s)
        if not (1 <= idx <= len(files)):
            raise ValueError
    except Exception:
        print("Invalid selection, defaulting to 1.")
        idx = 1

    chosen = files[idx - 1]
    return str(chosen.resolve())
