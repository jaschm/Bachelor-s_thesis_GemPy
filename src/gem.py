#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
from pathlib import Path
from contextlib import contextmanager

import numpy as np
import matplotlib.pyplot as plt
import gempy as gp
import gempy_viewer as gpv
import pyvista as pv
import trimesh
import ezdxf
import rasterio
from rasterio.crs import CRS
from rasterio.transform import from_origin
from scipy.interpolate import griddata
from gempy.core.data import StackRelationType

from main import main
from topography import set_topography

try:
    import pyvista._deprecate_positional_args as _dpa
    _dpa.warn_positional_args = lambda *a, **k: None
except Exception:
    pass

@contextmanager
def on_same_drive_as(module):
    """Vaihda tilapäisesti työskentelyasema samaksi kuin module.__file__ (esim. C:)."""
    cwd = Path.cwd()
    try:
        drive = Path(module.__file__).drive
        if drive:
            os.chdir(drive + "\\")
        yield
    finally:
        os.chdir(cwd)

config_path = Path("aineiston_kasittely/config_files/config.json")
with config_path.open("r", encoding="utf-8") as f:
    config = json.load(f)

def convert_obj_to_dxf(obj_path: str | Path, dxf_path: str | Path) -> None:
    obj_path = str(obj_path)
    dxf_path = str(dxf_path)
    mesh = trimesh.load(obj_path, force="mesh")
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError(f"OBJ ei ollut kolmioverkko: {obj_path}")
    doc = ezdxf.new()
    msp = doc.modelspace()
    for face in mesh.faces:
        pts = [mesh.vertices[i] for i in face]
        if len(pts) == 3:
            msp.add_3dface([pts[0], pts[1], pts[2]])
    Path(dxf_path).parent.mkdir(parents=True, exist_ok=True)
    doc.saveas(dxf_path)
    print(f"Saved DXF: {dxf_path}")

def convert_obj_to_tif(
    obj_path: str | Path,
    tif_path: str | Path,
    pixel_size: float = 2.0,
    nodata_val: float = -9999.0,
    crs_epsg: int = 3067,
) -> None:
    obj_path = str(obj_path)
    tif_path = str(tif_path)

    mesh = pv.read(obj_path)
    if mesh.n_points == 0:
        raise ValueError(f"Mesh has no points: {obj_path}")

    points = mesh.points
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    x_min, x_max = float(np.min(x)), float(np.max(x))
    y_min, y_max = float(np.min(y)), float(np.max(y))

    grid_x, grid_y = np.meshgrid(
        np.arange(x_min, x_max + pixel_size, pixel_size),
        np.arange(y_max, y_min - pixel_size, -pixel_size),
    )
    grid_z = griddata((x, y), z, (grid_x, grid_y), method="linear")
    grid_z_clean = np.where(np.isnan(grid_z), nodata_val, grid_z)

    transform = from_origin(x_min, y_max, pixel_size, pixel_size)
    Path(tif_path).parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        tif_path, "w",
        driver="GTiff",
        height=grid_z_clean.shape[0],
        width=grid_z_clean.shape[1],
        count=1,
        dtype="float32",
        crs=CRS.from_epsg(crs_epsg),
        transform=transform,
        nodata=nodata_val
    ) as dst:
        dst.write(grid_z_clean.astype("float32"), 1)

    print(f"Saved TIF: {tif_path}")

def export_dc_meshes_as_surfaces(geo_model, output_folder: str | Path = "exported_meshes") -> None:
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    names = [e.name for e in geo_model.structural_frame.structural_elements]

    for i, dc_mesh in enumerate(geo_model.solutions.dc_meshes):
        if dc_mesh is None:
            continue
        name_i = names[i] if i < len(names) else f"elem_{i}"
        if name_i.lower() in {"kallio", "basement"}:
            print(f"Skipping basement mesh at index {i} ({name_i}).")
            continue

        try:
            vertices_real = geo_model.input_transform.apply_inverse(dc_mesh.vertices)
            cloud = pv.PolyData(vertices_real)
            surface = cloud.delaunay_2d()

            if surface.n_points > 0:
                obj_path = output_folder / f"{name_i}_surface.obj"
                vtp_path = output_folder / f"{name_i}_surface.vtp"
                dxf_path = output_folder / f"{name_i}_surface.dxf"
                tif_path = output_folder / f"{name_i}_surface.tif"

                surface.save(str(obj_path))
                surface.save(str(vtp_path))
                print(f"Surface '{name_i}' saved as:\n  OBJ: {obj_path}\n  VTP: {vtp_path}")

                convert_obj_to_dxf(obj_path, dxf_path)
                convert_obj_to_tif(obj_path, tif_path)
            else:
                print(f"Surface '{name_i}' has no points — skipping.")
        except Exception as e:
            print(f"Error processing surface '{name_i}': {e}")

def gempy_main() -> None:
    bounds = main()
    if not bounds:
        print("No bounds returned from main.main(). Aborting GemPy build.")
        return
    x_min, x_max, y_min, y_max, z_min, z_max = bounds

    resolution_cfg = config.get("resolution", [30, 30, 30])
    if not (isinstance(resolution_cfg, (list, tuple)) and len(resolution_cfg) == 3):
        print(f"Invalid resolution in config.json ({resolution_cfg}), using default [30, 30, 30]")
        resolution_cfg = [30, 30, 30]

    geo_model = gp.create_geomodel(
        project_name="Luk-tutkielma",
        extent=[x_min, x_max, y_min, y_max, z_min, z_max],
        resolution=resolution_cfg,
        importer_helper=gp.data.ImporterHelper(
            path_to_orientations="aineiston_kasittely/output_data/orientation_offset.csv",
            path_to_surface_points="aineiston_kasittely/output_data/offset_data.csv",
        ),
    )

    gp.map_stack_to_surfaces(
        gempy_model=geo_model,
        mapping_object={
            "Postglasiaalinen": ("Sa",),
            "Glasiaalinen": ("Mr",),
            "basement": ("kallio",),
        },
    )

    sf = geo_model.structural_frame

    def _get_group_ci(frame, name):
        for g in frame.structural_groups:
            if g.name.lower() == name.lower():
                return g
        raise ValueError(f"group not found: {name}")

    for target_name, element_name in (("Postglasiaalinen", "Sa"),
                                      ("Glasiaalinen", "Mr")):
        try:
            g = _get_group_ci(sf, target_name)
            g.structural_relation = StackRelationType.ONLAP
            print(f"{g.name} -> ONLAP (by name)")
        except ValueError:
            set_ok = False
            for g in sf.structural_groups:
                if any(e.name == element_name for e in g.elements):
                    g.structural_relation = StackRelationType.ONLAP
                    print(f"{g.name} -> ONLAP (by element '{element_name}')")
                    set_ok = True
                    break
            if not set_ok:
                print(f"Could not set ONLAP for {target_name}/{element_name}")

    set_base = False
    for name in ("Kallioperä", "basement"):
        try:
            g = _get_group_ci(sf, name)
            g.structural_relation = StackRelationType.BASEMENT
            print(f"{g.name} -> BASEMENT (by name)")
            set_base = True
            break
        except ValueError:
            pass

    if not set_base:
        try:
            kallio_elem = sf.get_element_by_name("kallio")
            for g in sf.structural_groups:
                if kallio_elem in g.elements:
                    g.structural_relation = StackRelationType.BASEMENT
                    print(f"{g.name} -> BASEMENT (by element 'kallio')")
                    break
        except ValueError:
            print("Could not set BASEMENT: element 'kallio' not found")

    print("\n--- Structural frame ---")
    for g in sf.structural_groups:
        print("Group:", g.name, "| relation:", g.structural_relation, "| elements:", [e.name for e in g.elements])
    print("------------------------\n")

    hex_colormap = config.get("colormap", {})
    basement_hex = hex_colormap.get("basement") or hex_colormap.get("Basement")
    if basement_hex:
        sf.basement_color = basement_hex
    for elem in sf.structural_elements:
        if elem.name != "basement" and elem.name in hex_colormap:
            elem.color = hex_colormap[elem.name]

    print("\n--- Structural frame ---")
    for g in sf.structural_groups:
        print("Group:", g.name, "| relation:", g.structural_relation, "| elements:", [e.name for e in g.elements])
    print("------------------------\n")

    sections = {"section1": ([x_min, y_min], [x_max, y_max], [1, 50])}
    custom_sec = config.get("custom_section")
    if isinstance(custom_sec, dict):
        sec_name = str(custom_sec.get("name", "custom"))
        sec_start = list(custom_sec.get("start", [x_min, y_min]))
        sec_end = list(custom_sec.get("end", [x_max, y_max]))
        sec_samp = list(custom_sec.get("samples", [200, 100]))
        sections[sec_name] = (sec_start, sec_end, sec_samp)
    gp.set_section_grid(grid=geo_model.grid, section_dict=sections)

    set_topography(geo_model, x_min, x_max, y_min, y_max, z_min, downsample_factor=1)

    pv.global_theme.allow_empty_mesh = True
    gp.compute_model(geo_model)

    output_folder = Path("exported_meshes")
    output_folder.mkdir(parents=True, exist_ok=True)
    geo_model.surface_points_copy.df.to_csv(output_folder / "surface_points.csv", index=False)
    print("Exported surface points to CSV.")
    export_dc_meshes_as_surfaces(geo_model, output_folder)

    gpv.plot_2d(model=geo_model, ve=2, show_data=True, show_scalar=False, show_lith=True, show_topography=True)
    gpv.plot_2d(model=geo_model, ve=1, show_topography=True, show_data=True, legend=False)
    gpv.plot_2d(model=geo_model, ve=3, show_topography=True, show_data=True, legend=False)
    gpv.plot_2d(model=geo_model, ve=3, show_topography=True, show_data=True, legend=True, scalar_field=False)
    gpv.plot_2d(model=geo_model, ve=3, show_topography=True, show_data=False, legend=True, show_boundaries=False)
    gpv.plot_2d(model=geo_model, ve=3, show_topography=True, show_data=True, legend=True, show_boundaries=False)
    gpv.plot_2d(model=geo_model, ve=3, direction='x', show_topography=True, show_data=True, legend=True, show_boundaries=False)

    with on_same_drive_as(pv):
        gpv.plot_3d(geo_model, ve=1, show_topography=True, show_lith=True, image=False)
    with on_same_drive_as(pv):
        gpv.plot_3d(geo_model, ve=1, show_topography=True, show_lith=False, image=False)
    with on_same_drive_as(pv):
        gpv.plot_3d(geo_model, ve=1, show_topography=True, show_lith=True, image=False)
    with on_same_drive_as(pv):
        gpv.plot_3d(geo_model, ve=1, show_topography=False, show_lith=True, image=False)
    with on_same_drive_as(pv):
        gpv.plot_3d(geo_model, ve=1, show_topography=False, show_lith=False, image=False)
    with on_same_drive_as(pv):
        gpv.plot_3d(geo_model, ve=1, show_data=False, show_topography=False, show_lith=True, image=False)
    with on_same_drive_as(pv):
        gpv.plot_3d(geo_model, ve=1, show_data=False, show_topography=False, show_lith=False, image=False)
    with on_same_drive_as(pv):
        gpv.plot_3d(geo_model, ve=2, show_topography=False, show_lith=False, image=False)
    with on_same_drive_as(pv):
        gpv.plot_3d(geo_model, ve=3, show_topography=False, show_lith=False, image=False)

    plt.show()

if __name__ == "__main__":
    gempy_main()
