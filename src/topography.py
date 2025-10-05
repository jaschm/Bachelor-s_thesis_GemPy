import gempy as gp

import rasterio
from rasterio.windows import from_bounds
from rasterio.transform import Affine

def set_topography(geo_model, x_min, x_max, y_min, y_max, z_min, downsample_factor=1):
    input_path = 'aineiston_kasittely/input_data/L3324D.tif'
    output_path = 'aineiston_kasittely/input_data/L3324D_2_no_nodata.tif'

    with rasterio.open(input_path) as src:
        window = from_bounds(x_min, y_min, x_max, y_max, src.transform)
        transform = src.window_transform(window)
        band1 = src.read(1, window=window)

        nodata = src.nodata
        band1[band1 == nodata] = z_min

        if downsample_factor > 1:
            band1 = band1[::downsample_factor, ::downsample_factor]
            transform = Affine(
                transform.a * downsample_factor, transform.b, transform.c,
                transform.d, transform.e * downsample_factor, transform.f
            )

        profile = src.profile
        profile.update({
            'height': band1.shape[0],
            'width': band1.shape[1],
            'transform': transform,
            'nodata': z_min
        })

        with rasterio.open(output_path, 'w', **profile) as dst:
            dst.write(band1, 1)

    print(f"Setting topography with crop and downsample factor {downsample_factor}")
    gp.set_topography_from_file(
        grid=geo_model.grid,
        filepath=output_path
    )
