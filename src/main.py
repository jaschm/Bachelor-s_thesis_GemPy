import os

import json
import csv
import pandas as pd
import rasterio
import re
import ast

from utils.choose_input_file import choose_input_file
from data_processing.write_to_csv import write_to_csv
from utils.read_filtered_data import read_filtered_data
from utils.create_orientation_file import create_orientation_file

def parse_input(input_file):
    data = []
    with open('aineiston_kasittely/config_files/config.json', 'r', encoding='utf-8') as cfg_file:
        config = json.load(cfg_file)
    entry = config["entry"]
    start_keys = tuple(config["start_keys"])
    line_keys = config["line_keys"]
    alias_map_cfg = config.get("alias_map", {})
    alias_map = {k.lower(): v for k, v in alias_map_cfg.items()}

    numeric_pattern = re.compile(r"\b\d+\.\d{2}\b")
    last_line = ""
    alias_allowed = False

    with open(input_file, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            if line.startswith("PK"):
                continue

            if ' ' in line:
                if line.startswith(start_keys):
                    key, value = line.split(maxsplit=1)
                    entry[key] = value

                elif line.startswith("XY"):
                    parts = line.split()
                    entry['Y'], entry['X'], entry['Z'], entry['pvm'], entry['nro'] = parts[1:]

                elif line.startswith('-1'):
                    if entry['Paattymissyvyys'] == '':
                        numeric_value = re.findall(numeric_pattern, last_line)
                        if numeric_value:
                            entry['Paattymissyvyys'] = numeric_value[0]
                    data.append(entry.copy())
                    with open('aineiston_kasittely/config_files/config.json', 'r', encoding='utf-8') as cfg_file2:
                        entry = json.load(cfg_file2)["entry"]
                    alias_allowed = False

                elif any(col in line for col in line_keys):
                    parts = line.split()
                    if len(parts) >= 2:
                        raw_name = parts[-1].strip()
                        raw_name_cap = raw_name.capitalize()

                        if raw_name_cap == 'Sa':
                            alias_allowed = True

                        name_effective = raw_name_cap
                        if alias_allowed:
                            name_effective = alias_map.get(raw_name.lower(), raw_name_cap)

                        column_name = name_effective.capitalize()
                        if column_name in entry:
                            if isinstance(entry[column_name], list):
                                entry[column_name].append(parts[0])
                            else:
                                entry[column_name] = [entry[column_name], parts[0]]
                        elif column_name == 'Paattymissyvyys':
                            numeric_value = re.findall(numeric_pattern, line)
                            if numeric_value:
                                entry[column_name] = numeric_value[0]

            last_line = line

    return data


def main():
    input_directory = "aineiston_kasittely/input_data"
    input_file = choose_input_file(input_directory)
    if not input_file:
        print("→ No valid input file selected. Exiting.")
        return None

    print(f"Selected file: {input_file}")
    output_file = "aineiston_kasittely/output_data/output_tek_to_csv.csv"
    data = parse_input(input_file)
    write_to_csv(data, output_file)
    print(f"→ Rows parsed from .tek → CSV: {len(data)}")

    with open('aineiston_kasittely/config_files/config.json', 'r', encoding='utf-8') as cfg_file:
        config = json.load(cfg_file)
        layers_to_include = set(config.get("layers_to_include", []))
        layers_to_ignore = set(config.get("layers_to_ignore", []))

    if layers_to_include and 'kallio' in layers_to_include and 'Mr' not in layers_to_include:
        layers_to_include = set(layers_to_include)
        layers_to_include.add('Mr')

    formation_cols = ['Sa', 'Mr', 'Ka', 'Sr', 'SR', 'Sasi', 'Srmr', 'liSa', 'Ki']
    alias_map = {k.lower(): v for k, v in config.get("alias_map", {}).items()}

    skip_tt_values = {"PO - 0", "NO 1 - - -", "SI  0 0"}
    offset_rows = []
    boreholes_total = 0
    boreholes_used = 0

    with open(output_file, mode='r', newline='', encoding='utf-8') as infile:
        reader = csv.DictReader(infile)
        for row in reader:
            boreholes_total += 1
            if row.get("TT") in skip_tt_values:
                continue

            try:
                z_value = float(row['Z'])
                paattymis = float(row['Paattymissyvyys'])
                x = float(row['X'])
                y = float(row['Y'])
                nro = row['nro']
            except Exception as e:
                print(f"→ Skipping borehole {row.get('nro', '?')} due to parse error: {e}")
                continue

            all_layers = []
            alias_allowed = False
            found_any = False

            for col in formation_cols:
                field = row.get(col)
                if isinstance(field, str) and '[' in field:
                    try:
                        depths = ast.literal_eval(field)
                        cleaned = sorted({float(d) for d in depths if str(d).strip() != ""})
                        if cleaned:
                            if col.capitalize() == 'Sa':
                                alias_allowed = True
                            name = col.capitalize()
                            if alias_allowed:
                                name = alias_map.get(col.lower(), col).capitalize()

                            if layers_to_include:
                                if name not in layers_to_include:
                                    continue
                            elif name in layers_to_ignore:
                                continue

                            all_layers.append((name, cleaned))
                            found_any = True
                    except Exception:
                        continue

            if not found_any:
                continue

            all_layers.sort(key=lambda t: min(t[1]))

            for i, (formation, depths) in enumerate(all_layers):
                if not depths:
                    continue
                top_depth = min(depths)
                if i < len(all_layers) - 1:
                    next_top_depth = min(all_layers[i + 1][1])
                    bottom_depth = next_top_depth
                else:
                    bottom_depth = paattymis

                bottom_z = round(z_value - bottom_depth, 3)

                if i == len(all_layers) - 2:
                    bottom_z += 0.1

                offset_rows.append({
                    'piste': nro,
                    'X': x,
                    'Y': y,
                    'Z': bottom_z,
                    'formation': formation
                })
            boreholes_used += 1

    print(f"→ Boreholes total: {boreholes_total} | used: {boreholes_used}")
    print(f"→ Offset rows (from boreholes) so far: {len(offset_rows)}")

    if not offset_rows:
        print("→ No offset rows produced from boreholes. "
              "Check 'layers_to_include', aliasing (SR→Mr), and TT filters.")
        return None

    x_vals = [row['X'] for row in offset_rows]
    y_vals = [row['Y'] for row in offset_rows]
    x_min, x_max = min(x_vals), max(x_vals)
    y_min, y_max = min(y_vals), max(y_vals)

    ext_added = 0
    for ext_file, formation in [
        ('kallio_surface_points_combined.csv', 'Mr'),
    ]:
        try:
            path = f'aineiston_kasittely/output_data/{ext_file}'
            df_ext = pd.read_csv(path)
            df_ext.columns = df_ext.columns.str.strip()
            df_ext = df_ext[(df_ext['X'] >= x_min) & (df_ext['X'] <= x_max) &
                            (df_ext['Y'] >= y_min) & (df_ext['Y'] <= y_max)]

            print(f"-> Filtered {formation}_ext points from {ext_file}: {len(df_ext)} rows")

            if (layers_to_include and formation in layers_to_include) or \
               (not layers_to_include and formation not in layers_to_ignore):
                for i, r in df_ext.iterrows():
                    offset_rows.append({
                        'piste': 20000 + i,
                        'X': r['X'],
                        'Y': r['Y'],
                        'Z': r['Z'],
                        'formation': formation
                    })
                    ext_added += 1
        except FileNotFoundError:
            print(f"→ Optional external file not found: {ext_file} (skipping)")
        except Exception as e:
            print(f"→ Failed to add {formation} surface points from {ext_file}: {e}")

    print(f"→ External points added: {ext_added}")
    if not offset_rows:
        print("→ Still no offset rows after external points. Cannot proceed.")
        return None

    df_offset = pd.DataFrame(offset_rows)
    df_offset = df_offset.dropna(subset=['Z'])

    df_offset['formation'] = df_offset['formation'].str.strip()
    prio = {'Sa': 0, 'Mr': 1}
    df_offset['formation_priority'] = df_offset['formation'].map(prio).fillna(2).astype(int)
    df_offset.sort_values(by=['formation_priority', 'formation', 'piste', 'Z'],
                          ascending=[True, True, True, False], inplace=True)
    df_offset.drop(columns='formation_priority', inplace=True)

    offset_path = 'aineiston_kasittely/output_data/offset_data.csv'
    df_offset.to_csv(offset_path, index=False)
    print(f"\n → offset_data.csv written. Total rows: {len(df_offset)}\n")

    offset_points = read_filtered_data(offset_path)
    orientation_file = 'aineiston_kasittely/output_data/orientation_offset.csv'
    create_orientation_file(offset_points, orientation_file)
    print(f"→ Orientation file created: {orientation_file}")

    x_min, x_max = df_offset['X'].min(), df_offset['X'].max()
    y_min, y_max = df_offset['Y'].min(), df_offset['Y'].max()
    z_min_points, z_max_points = df_offset['Z'].min(), df_offset['Z'].max()

    with rasterio.open('aineiston_kasittely/input_data/L3324D.tif') as dataset:
        band1 = dataset.read(1)
        nodata = dataset.nodata
        valid_data = band1[band1 != nodata]
        z_min_dtm = float(valid_data.min())
        z_max_dtm = float(valid_data.max())

        x_min = max(float(x_min), float(dataset.bounds.left))
        x_max = min(float(x_max), float(dataset.bounds.right))
        y_min = max(float(y_min), float(dataset.bounds.bottom))
        y_max = min(float(y_max), float(dataset.bounds.top))

    z_min = min(float(z_min_points), z_min_dtm)
    z_max = max(float(z_max_points), z_max_dtm)

    with rasterio.open('aineiston_kasittely/input_data/L3324D_2_no_nodata.tif') as dataset2:
        band1 =  dataset2.read(1)
        nodata = dataset2.nodata
        valid_data = band1[band1 != nodata]
        z_min_dtm = float(valid_data.min())
        z_max_dtm = float(valid_data.max())

    z_min = min(float(z_min_points), z_min_dtm)
    z_max = max(float(z_max_points), z_max_dtm)
    
    if x_min >= x_max or y_min >= y_max:
        print(f"Invalid grid size after clipping to DTM "
              f"(x_min={x_min}, x_max={x_max}, y_min={y_min}, y_max={y_max}).")
        return None

    print(f"Laaujuus → X: {x_min}..{x_max} | Y: {y_min}..{y_max} | Z: {z_min}..{z_max}")
    return x_min, x_max, y_min, y_max, z_min, z_max

if __name__ == "__main__":
    main()
