"""
This file contains all utility functions needed for the extreme event analysis
"""
import os.path
import numpy as np
from shapely.geometry import shape
import xarray as xr
from pyproj import Geod
from settings import SURFACE_AREA_PATH

# define resolution and boundary points of UNIFORM ISIMIP lattice
ISIMIP_RESOLUTION = 0.5
LAT_MIN = -89.75
LAT_MAX = 89.75
LON_MIN = -179.75
LON_MAX = 179.75
# specify a named ellipsoid
geod = Geod(ellps="WGS84")

# define ISIMIP name dictionary
ISIMIP_IMPACT_NAME = {
    "cropfailedarea": "Crop failure",
    "heatwavedarea": "Heatwave",
    "burntarea": "Wildfire",
    "driedarea": "Drought",
    "floodedarea": "Flood",
}
ISIMIP_IMPACT_LABEL = {
    "cropfailedarea": "(a)",
    "heatwavedarea": "(b)",
    "burntarea": "(c)",
}
ISMIP_GCM_COLOR = {
    "gfdl-esm4": "tab:red",
    "ukesm1-0-ll": "tab:blue",
    "ipsl-cm6a-lr": "tab:green",
    "mpi-esm1-2-hr": "tab:orange",
    "mri-esm2-0": "tab:purple",
    "gswp3-w5e5": "tab:red",
    "20crv3-era5": "tab:blue",
    "20crv3-w5e5": "tab:green",
    "20crv3": "tab:orange",
}
CROP_NAMES = {
    ("mai", "firr"): "maize_irrigated",
    ("mai", "noirr"): "maize_rainfed",
    ("ri1", "firr"): "rice_irrigated",
    ("ri2", "firr"): "rice_irrigated",
    ("ri1", "noirr"): "rice_rainfed",
    ("ri2", "noirr"): "rice_rainfed",
    ("soy", "firr"): "oil_crops_soybean_irrigated",
    ("soy", "noirr"): "oil_crops_soybean_rainfed",
    ("swh", "firr"): "temperate_cereals_irrigated",
    ("wwh", "firr"): "temperate_cereals_irrigated",
    ("swh", "noirr"): "temperate_cereals_rainfed",
    ("wwh", "noirr"): "temperate_cereals_rainfed",
}

# if gridded surface area file does not exist compute it
if os.path.exists(SURFACE_AREA_PATH):
    surface_area = xr.open_dataarray(SURFACE_AREA_PATH)
else:
    # set up shapes for each square in lon lat grid
    surface_area = np.array(
        [
            [
                shape(
                    {
                        "type": "Polygon",
                        "coordinates": [
                            [
                                [
                                    lon - ISIMIP_RESOLUTION / 2,
                                    lat - ISIMIP_RESOLUTION / 2,
                                ],
                                [
                                    lon + ISIMIP_RESOLUTION / 2,
                                    lat - ISIMIP_RESOLUTION / 2,
                                ],
                                [
                                    lon + ISIMIP_RESOLUTION / 2,
                                    lat + ISIMIP_RESOLUTION / 2,
                                ],
                                [
                                    lon - ISIMIP_RESOLUTION / 2,
                                    lat + ISIMIP_RESOLUTION / 2,
                                ],
                                [
                                    lon - ISIMIP_RESOLUTION / 2,
                                    lat - ISIMIP_RESOLUTION / 2,
                                ],
                            ]
                        ],
                    }
                )
                for lat in np.arange(
                    LAT_MIN, LAT_MAX + ISIMIP_RESOLUTION, ISIMIP_RESOLUTION
                )
            ]
            for lon in np.arange(
                LON_MIN, LON_MAX + ISIMIP_RESOLUTION, ISIMIP_RESOLUTION
            )
        ]
    )
    # calculate geodesy area and convert from m^2 to km^2
    surface_area = (
        np.array(
            [abs(geod.geometry_area_perimeter(x)[0]) for x in surface_area.flatten()]
        ).reshape(surface_area.shape)
        / 10.0**6
    )
    # convert to DataArray
    surface_area = xr.DataArray(
        data=surface_area,
        dims=["lon", "lat"],
        coords={
            "lat": np.arange(LAT_MIN, LAT_MAX + ISIMIP_RESOLUTION, ISIMIP_RESOLUTION),
            "lon": np.arange(LON_MIN, LON_MAX + ISIMIP_RESOLUTION, ISIMIP_RESOLUTION),
        },
        attrs={"standard_name": "area of grid cell", "unit": "km^2"},
    )
    surface_area.to_netcdf(SURFACE_AREA_PATH)
