import os
import re
import sys
import glob
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import kairauksen_paattyminen

def resolve_input_dir():
    here = os.path.abspath(os.path.dirname(__file__))
    candidate = os.path.join(here, "aineiston_kasittely", "input_data")
    if os.path.isdir(candidate):
        return candidate
    cwd_candidate = os.path.join(os.getcwd(), "aineiston_kasittely", "input_data")
    if os.path.isdir(cwd_candidate):
        return cwd_candidate
    return os.getcwd()

def list_input_files(input_dir, patterns=(".tek", ".txt")):
    
    results = []
    for ext in patterns:
        results.extend(glob.glob(os.path.join(input_dir, f"*{ext}")))
    results = sorted(set(results), key=lambda p: os.path.basename(p).lower())
    return results

def choose_file_from_dir(input_dir):
  
    files = list_input_files(input_dir)
    if not files:
        print(f"[VIRHE] Kansiossa ei ole .tek/.txt -tiedostoja: {input_dir}")
        return None

    print("\nValitse luettava tiedosto:")
    for i, path in enumerate(files, start=1):
        try:
            stat = os.stat(path)
            size_kb = stat.st_size / 1024.0
            mtime = datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M")
            print(f"  {i:2d}) {os.path.basename(path)}  ({size_kb:.1f} kB, muokattu {mtime})")
        except OSError:
            print(f"  {i:2d}) {os.path.basename(path)}")

    while True:
        choice = input("Anna tiedoston numero (tai tyhjä peruuttaaksesi): ").strip()
        if choice == "":
            return None
        if choice.isdigit():
            idx = int(choice)
            if 1 <= idx <= len(files):
                return files[idx - 1]
        print("Virheellinen valinta. Yritä uudelleen.")

def parse_termination_code(line: str):
    """
    Matches real-world variants:
      - "-1 KL", "–1 KL", "—1 KL", "-1,KL", "-1  kl  comment"
    Returns e.g. "KL" or None.
    """
    if not line:
        return None
    s = line.strip()
    s = s.replace("\u2013", "-").replace("\u2014", "-").replace("\u2212", "-")
    if s.startswith("-1"):
        rest = s[2:].lstrip(" \t,;:")
        m = re.match(r"([A-Za-zÅÄÖåäö]{1,3})", rest, flags=re.IGNORECASE)
        return m.group(1).upper() if m else None
    m = re.match(r"^\s*-\s*1\s*[,;:]?\s*([A-Za-zÅÄÖåäö]{1,3})\b", s, flags=re.IGNORECASE)
    return m.group(1).upper() if m else None

def parse_tek_file(file_path, project_number, point_number):
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    d = {
        "depth": [], "weight": [], "half_turns": [], "soil_type": [],
        "Y": None, "X": None, "Z": None, "date": None, "point": None,
        "termination_type": "-", "Or": "-", "Tyonro": "-"
    }
    capture_project = False
    capture_point = False
    current_soil_type = "-"

    for raw in lines:
        line = raw.strip()
        if not line:
            continue

        if line.startswith("TY"):
            capture_project = (project_number in line)
            d["Tyonro"] = line[2:].strip() or project_number
            if not capture_project:
                capture_point = False
            continue

        if capture_project and (line.startswith("OR") or line.startswith("Or")):
            parts = line.split()
            if len(parts) >= 2:
                d["Or"] = parts[1]
            continue

        if capture_project and line.startswith("XY"):
            parts = line.split()
            if len(parts) >= 6:
                try:
                    d["Y"] = float(parts[1].replace(",", "."))
                    d["X"] = float(parts[2].replace(",", "."))
                    d["Z"] = float(parts[3].replace(",", "."))
                except ValueError:
                    pass
                d["date"] = parts[4]
                d["point"] = parts[5]
                capture_point = (d["point"] == point_number)
            else:
                capture_point = False
            continue

        if not (capture_project and capture_point):
            continue

        code = parse_termination_code(line)
        if code:
            d["termination_type"] = code
            break

        if line.startswith("AL") or line[0].isdigit() or (line[0] == "-" and len(line) > 1 and line[1].isdigit()):
            parts = line.split()
            tokens = parts[1:] if parts[0] == "AL" else parts
            try:
                d["depth"].append(float(tokens[0].replace(",", ".")))
                d["weight"].append(int(float(tokens[1])))
                d["half_turns"].append(int(float(tokens[2])))
                if len(tokens) > 3 and tokens[3]:
                    current_soil_type = tokens[3]
                d["soil_type"].append(current_soil_type or "-")
            except (ValueError, IndexError):
                continue

    max_len = max(len(d["depth"]), len(d["weight"]), len(d["half_turns"]), len(d["soil_type"])) if d["depth"] else 0
    for key in ["depth", "weight", "half_turns", "soil_type"]:
        while len(d[key]) < max_len:
            d[key].append("-" if key == "soil_type" else 0)

    return d

def list_projects_and_points(file_path):
    projects = {}
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    current_project = None
    for line in lines:
        if line.startswith("TY"):
            current_project = line.strip()
            projects[current_project] = []
        if line.startswith("XY") and current_project:
            projects[current_project].append(line.strip())
    return projects

def create_m_with_shorter_middle(ax, x_left, x_right, y_bottom, y_top):
    width = x_right - x_left
    m_width_scale = 0.6
    m_height_scale = 0.6
    middle_valley_scale = 0.8
    m_points = [
        (x_left + width * (1 - m_width_scale) / 2, y_top - (y_top - y_bottom) * (1 - m_height_scale)),
        (x_left + width * (1 - m_width_scale) / 2 + width * m_width_scale * 0.3, y_bottom),
        (x_left + width * (1 - m_width_scale) / 2 + width * m_width_scale * 0.5, y_bottom + (y_top - y_bottom) * (1 - middle_valley_scale)),
        (x_left + width * (1 - m_width_scale) / 2 + width * m_width_scale * 0.7, y_bottom),
        (x_left + width * (1 - m_width_scale) / 2 + width * m_width_scale, y_top - (y_top - y_bottom) * (1 - m_height_scale))
    ]
    xs, ys = zip(*m_points)
    ax.plot(xs, ys, color="brown", linewidth=0.7)

def normalize_for_pretty_plot(parsed):
    depth = parsed.get("depth", []) or []
    weight = parsed.get("weight", []) or []
    half_turns = parsed.get("half_turns", []) or []
    soil_type = parsed.get("soil_type", []) or []

    Z  = [float(parsed.get("Z", 0.0) or 0.0)]
    XY = [float(parsed.get("Y", 0.0) or 0.0), float(parsed.get("X", 0.0) or 0.0)]
    point  = [str(parsed.get("point", "-") or "-")]
    Or     = [str(parsed.get("Or", "-") or "-")]
    Tyonro = [str(parsed.get("Tyonro", "-") or "-")]
    Paattyminen = [str(parsed.get("termination_type", "-") or "-").upper()]

    return {
        "depth": depth,
        "weight": weight,
        "half_turns": half_turns,
        "soil_type": soil_type,
        "Z": Z,
        "XY": XY,
        "point": point,
        "Or": Or,
        "Tyonro": Tyonro,
        "Päättyminen": Paattyminen,
        "date": parsed.get("date", "-"),
        "Y": parsed.get("Y", 0.0),
        "X": parsed.get("X", 0.0),
    }

def get_termination_code(d):
    v = None
    if "Päättyminen" in d and isinstance(d["Päättyminen"], list) and d["Päättyminen"]:
        v = d["Päättyminen"][0]
    if not v:
        v = d.get("termination_type")
    return (str(v).strip().upper() if v is not None else "-")

def plot_corrected_z_title(tek_data, maalajipylvas_width=0.15):
    depth = tek_data["depth"]
    if not depth:
        print("No data rows for the selected point.")
        return

    weight = [w / 100 for w in tek_data["weight"]]
    half_turns = tek_data["half_turns"]
    soil_types = [s.title() if isinstance(s, str) else s for s in tek_data["soil_type"]]

    z_value = tek_data["Z"][0]
    point1  = tek_data["point"][0]
    org     = tek_data["Or"][0]
    tyonro  = tek_data["Tyonro"][0]
    termination_type = get_termination_code(tek_data)

    date = tek_data.get("date", "-")
    Y = tek_data.get("Y", 0.0)
    X = tek_data.get("X", 0.0)

    center = 0.0
    left  = center - maalajipylvas_width / 2
    right = center + maalajipylvas_width / 2

    fig, ax = plt.subplots(figsize=(10, 8))

    y_min, y_max = min(depth), max(depth)
    top_pad, bot_pad = 1.0, 0.7
    y_top = y_min - top_pad
    y_bottom = y_max + bot_pad
    ax.set_xlim(-1, 5)
    ax.set_ylim(y_bottom, y_top)

    ax.add_patch(
        patches.Rectangle(
            (left, y_min),
            maalajipylvas_width,
            y_max - y_min,
            fill=False, edgecolor="black", linewidth=0.25
        )
    )

    ax.plot([left, left],   [y_min, y_max], color="black", linewidth=0.5, zorder=2)
    ax.plot([right, right], [y_min, y_max], color="black", linewidth=0.5, zorder=2)

    for i in range(len(depth) - 1):
        symbol = ""
        color = "black"
        if soil_types[i] == "Mr":
            create_m_with_shorter_middle(ax, left, right, depth[i], depth[i + 1])
        elif soil_types[i] == "Sa":
            symbol, color = "|", "#64f6f4"
        elif soil_types[i] == "Si":
            symbol, color = "||", "purple"
        elif soil_types[i] == "Sr":
            symbol, color = "o o", "green"
        elif soil_types[i] == "Hk":
            symbol, color = ". .", "#F0EA52"
        elif soil_types[i] == "Ki":
            symbol, color = "▲", "black"
        if symbol:
            ax.text((left + right) / 2, (depth[i] + depth[i + 1]) / 2,
                    symbol, fontsize=10, ha="center", color=color)

    last_value = None
    for i in range(len(depth) - 1):
        ht = half_turns[i]
        if ht is None or ht > 100:
            continue
        if ht == 0:
            start_x = last_value if last_value is not None else left
            x = left - weight[i]
            ax.plot([start_x, x], [depth[i], depth[i]], color="black", linewidth=0.5, zorder=3)
            ax.plot([x, x], [depth[i], depth[i + 1]], color="black", linewidth=0.5, zorder=3)
            last_value = x
        elif ht > 0:
            start_x = last_value if last_value is not None else right
            x = right + ht / 100.0
            ax.plot([start_x, x], [depth[i], depth[i]], color="black", linewidth=0.5, zorder=3)
            ax.plot([x, x], [depth[i], depth[i + 1]], color="black", linewidth=0.5, zorder=3)
            last_value = x

    for s in ("top", "right", "left", "bottom"):
        ax.spines[s].set_visible(False)
    ax.grid(False)
    ax.set_ylabel("Syvyys (m)", fontsize=16)

    xticks = [left, left - 1, 0, right, right + 0.2, right + 0.4, right + 0.6, right + 0.8, right + 1.0]
    xtick_labels = ["0", "1 kN", "", "0", "20", "40", "60", "80", "100 pk/20cm"]
    ax.set_xticks(xticks)
    ax.set_xticklabels(xtick_labels, fontsize=8)

    plt.text(-1.5, -0.2, "Paino", fontsize=16, ha="left", va="center")
    plt.text( 0.5, -0.2, "Kierto", fontsize=16, ha="left", va="center")

    plt.text(
        -1.5, y_min - 0.6,
        f"Työn nro: {tyonro} Piste: {point1}",
        fontsize=16, ha="left", va="bottom"
    )

    draw_map = {
        "TM": kairauksen_paattyminen.kairaus_paattynyt_tiiviiseen_maakerrokseen,
        "KI": kairauksen_paattyminen.kairaus_paattynyt_kiveen_tai_lohkareeseen,
        "KL": kairauksen_paattyminen.kairaus_paattynyt_kiveen_lohkareeseen_tai_kallioon,
        "KA": kairauksen_paattyminen.kairaus_paattynyt_kallioon_varmistettu_kallio,
        "MS": kairauksen_paattyminen.kairaus_paattynyt_maarasyvyyteen,
        "KN": kairauksen_paattyminen.kairaus_paattynyt_kiilautumalla_kivien_tai_lohkareiden_valiin,
    }
    y_bottom_symbol = y_max - 0.6
    fn = draw_map.get(termination_type)
    if fn:
        fn(ax, x_center=0, y_bottom=y_bottom_symbol)
    else:
        print(f"[WARN] Unknown termination code '{termination_type}', no symbol drawn.")

    ax.text(-0.35, -0.15, f"{'+' if z_value >= 0 else ''}{z_value:.3f}", fontsize=16, ha="left", va="center")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    input_dir = resolve_input_dir()
    print(f"Käytettävä kansio: {input_dir}")
    tek_file = choose_file_from_dir(input_dir)

    if not tek_file:
        print("Tiedoston valinta peruutettu.")
        sys.exit(0)

    if os.path.exists(tek_file):
        projects = list_projects_and_points(tek_file)
        if not projects:
            print("[INFO] Tiedostosta ei löytynyt TY/XY-rivejä.")
        else:
            print("\nSaatavilla olevat projektit ja XY-rivien määrät:")
            for project, points in projects.items():
                print(f"  {project}: {len(points)} kpl")

        project_number = input("\nAnna projektinumero: ").strip()
        point_number   = input("Anna pisteen tunnus: ").strip()

        parsed = parse_tek_file(tek_file, project_number, point_number)
        print(f"[INFO] Päättymiskoodi havaittu: {parsed.get('termination_type')}")
        tek_data = normalize_for_pretty_plot(parsed)
        plot_corrected_z_title(tek_data)
    else:
        print("Annettu TEK-tiedosto ei ole olemassa:", tek_file)
