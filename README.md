# Bachelor's Thesis: **[3D modeling of Quaternary sediments in Littoinen, Kaarina, using the open-source GemPy library](https://urn.fi/URN:NBN:fi-fe20251016101769)**

This repository contains the full code and data structure for a bachelor's thesis project using the open-source **[GemPy](https://www.gempy.org)** library for 3D geological modelling.  
The example dataset models the subsurface of **Littoinen, Kaarina (Southwest Finland)** based on **[GTK ground investigation data](https://gtkdata.gtk.fi/pohjatutkimukset/index.html)**.

---

## ⚙️ Requirements

Create and activate a Python environment (e.g. Conda or venv) and install dependencies:

pip install gempy gempy-viewer pyvista vtk trimesh ezdxf rasterio matplotlib scipy numpy pandas

---

### Usage

Download the DEM file

Download elevation model: L3342D.tif from the National Land Survey of Finland File Service:
**[National Land Survey of Finland MapSite](https://asiointi.maanmittauslaitos.fi/karttapaikka/tiedostopalvelu/korkeusmalli?lang=fi)**

Insert the file into the input folder. Place the downloaded file into: aineiston_kasittely/input_data/L3342D.tif

Run the model script

Execute the following command in the project root:

python src/gem.py

Optional: Use other locations  
If you want to build a model from another location, download a TEK file from:  **[GTK ground investigation register](https://asiointi.maanmittauslaitos.fi/karttapaikka/tiedostopalvelu/korkeusmalli?lang=fi)** or from your own site’s Geotechnical investigator.

Note:  
In this case, you may also need to download a new DEM file for the corresponding area.  
You might also have to adjust some model parameters - some are defined in the config.json file, while others are hardcoded in the source code.
