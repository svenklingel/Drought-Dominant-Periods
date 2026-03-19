# pylint: disable = too-many-lines
"""
This script creates plots of the time convolutions of yearly ISIMIP extreme event time series data
and counts the number of extreme events in time bins as stored in OUTPUT_PATH
For more details check gitlab project.

Author: Karim Zantout
"""

import argparse
import glob
import os
import csv
import time
from datetime import datetime
import logging
from typing import Union
import sys
import cartopy.crs as ccrs
import pandas as pd
import pytz
import numpy as np
import xarray as xr
import geopandas as gpd
import matplotlib
from matplotlib import pyplot as plt
from matplotlib.cm import ScalarMappable
import cmcrameri.cm as cmc
from settings import (
    R2_THRESHOLD,
    USE_AR_MODEL,
    USE_ALL_GCM_MODELS,
    USE_ALL_IMP_MODELS,
    ALL_SSP_SCENARIOS,
    ALL_GCM_MODELS,
    ALL_IMPACT_MODELS,
    USE_MODEL_MEAN,
    USE_RESULT_MEDIAN,
    OUTPUT_PATH,
    LOG_PATH,
    LOCATIONS,
    t_0s,
    NT,
    PLOTS_PATH,
    SURFACE_AREA_PATH,
    RUN_CROP_TYPE_RESOLVED,
)
from util import (
    ISIMIP_IMPACT_NAME,
    ISMIP_GCM_COLOR,
    ISIMIP_IMPACT_LABEL,
    ISIMIP_IMPACT_PICONTROL_LABEL,
    FIG_LABEL_SINGLECROP,
    FOURIER_MARKERSTYLE
)
from main import TimeAnalysisImpacts


# pylint: disable=too-many-instance-attributes
class PlotTimeAnalysisImpacts(TimeAnalysisImpacts):
    """This class contains all plots of the extreme event time analysis"""

    # pylint: disable=dangerous-default-value
    def __init__(
        self,
        log: logging,
        ssp: list,
        impact_type: str,
        use_all_gcms: bool = USE_ALL_GCM_MODELS,
        use_all_imps: bool = USE_ALL_IMP_MODELS,
        use_mean: bool = USE_MODEL_MEAN,
        use_result_median: bool = USE_RESULT_MEDIAN,
        is_r2test: bool = False,
        wildfire_cap: float = -999.,
        crop_type: str = "",
    ):
        self.is_r2test = is_r2test
        self.wildfire_cap = wildfire_cap
        self.total_model_count_median = None
        self.total_model_count_min = None
        self.total_model_count_max = None
        self.event_counts = None
        self.use_result_median = use_result_median
        self.log = log
        if not self.use_result_median:
            print("Not using result median")
            # get all properties and methods of TimeAnalysisImpacts class
            super().__init__(
                log, ssp, impact_type, use_all_gcms, use_all_imps, use_mean, crop_type
            )
            super().calculate_local_dominant_return_period(True)
            super().calculate_local_dominant_return_period(False)
            super().count_impacts()
        else:
            self.ssp = ssp
            self.crop_type = crop_type
            self.ssp_name = None
            super().get_ssp_scenario_name()
            self.impact_model = ["median_median"]
            self.use_model_mean = False
            self.impact_type = impact_type
        # fix data input string
        self.data_path = OUTPUT_PATH
        if not self.use_result_median:
            if self.impact_time_series is not None:
                # read dominant frequency results for specific model combination
                self.read_dominant_return_period(no_trend=True)
                self.read_dominant_return_period(no_trend=False)
            # convert dict of xarrays into pandas
            for key, val in self.total_count.items():
                self.total_count[key] = pd.DataFrame.from_dict(
                    val, orient="index", dtype=float
                )
                self.total_count[key].index.name = "year"
                self.total_count[key] = (
                    self.total_count[key].rename(columns={0: "counts"}).reset_index()
                )
            # convert dict of data arrays into data set
            for key, val in self.impact_count_t0.items():
                self.impact_count_t0[key] = xr.Dataset(val)
        else:
            # read dominant frequency results for result median
            self.read_median_results("dominant_return_period", True)
            self.read_median_results("dominant_return_period", False)
            self.read_std_results("dominant_return_period", True)
            self.read_std_results("dominant_return_period", False)
            self.read_modelcount_results(True)
            self.read_modelcount_results(False)
            self.read_median_results("event_counts", False)
            self.read_std_results("event_counts", False)
            self.read_median_total_counts()

    def read_median_total_counts(
        self,
    ):
        """Read total event counts as csv"""
        self.total_count = {}
        self.total_model_count_median = {}
        self.total_model_count_min = {}
        self.total_model_count_max = {}
        filenames = glob.glob(
            os.path.join(OUTPUT_PATH, "event_counts", self.impact_type) + "/*.csv"
        )
        self.total_count[self.impact_type] = {}
        for filename in filenames:
            if not filename.endswith(".csv"):
                self.log.error(
                    "File name extension is expected to be .csv "
                    f"but got {filename.split('.')[-1]} instead"
                )
                raise ValueError
            # get ssp and n_t from filename
            # remove filetype ending and split
            _filename = os.path.basename(filename).split(".")[0].split("_")
            # check if Nt and ssp match:
            if (
                _filename[-3] not in ["median", "min", "max"]
                or int(_filename[-1].split("Nt")[1]) != NT
                or _filename[-2] != self.ssp_name
            ):
                continue
            if str(self.ssp_name) not in filename:
                continue
            self.log.info(f"Reading file {filename}...")
            # append data into a data set for each extreme event
            if _filename[-4] != self.impact_type and _filename[-3] == "median":
                self.total_model_count_median[
                    (self.impact_type, _filename[-4])
                ] = pd.read_csv(filename)
            elif _filename[-4] != self.impact_type and _filename[-3] == "min":
                self.total_model_count_min[
                    (self.impact_type, _filename[-4])
                ] = pd.read_csv(filename)
            elif _filename[-4] != self.impact_type and _filename[-3] == "max":
                self.total_model_count_max[
                    (self.impact_type, _filename[-4])
                ] = pd.read_csv(filename)
            elif _filename[-3] == "median":
                self.total_count[self.impact_type] = pd.read_csv(filename)
            self.log.info(f"{filename} successfully parsed!\n")
        self.log.info("Successfully parsed total area counts!")

    def read_dominant_return_period(self, no_trend: bool):
        """Read dominant frequencies as netcdf"""
        if not no_trend:
            self.dominant_return_period_t0 = {}
            container = self.dominant_return_period_t0
            output_subpath = "original"
        else:
            self.dominant_return_period_t0_no_trend = {}
            container = self.dominant_return_period_t0_no_trend
            output_subpath = "detrended"
        for event in self.impact_time_series.keys():
            self.log.info(
                f"Reading {self.impact_type}-{event} event dominant return period to file..."
            )
            full_path = os.path.join(
                OUTPUT_PATH,
                output_subpath,
                "dominant_return_period",
                f"{self.impact_type}_{self.impact_type}",
                f"{event.split('_')[0]}_{event.split('_')[0]}",
            )
            # read impact count data as netcdf
            file_path = os.path.join(
                full_path,
                f"{self.impact_type}_{event.split('_')[1]}_"
                f"{self.impact_type}_{event.split('_')[1]}_"
                f"{self.ssp_name}_extreme_event_Nt{NT}_dominant_frequency.nc",
            )
            if os.path.exists(file_path):
                container[(event, event)] = xr.open_dataset(file_path)
                # make sure that data variables are t_0 (int)
                container[(event, event)] = container[(event, event)].rename_vars(
                    {var: int(var) for var in container[(event, event)].data_vars}
                )
            else:
                self.log.info(
                    f"Skipping {file_path} because output file does not exist!"
                )
                continue
            self.log.info(
                f"{self.impact_type}-{event} event dominant return period "
                "successfully stored to file!"
            )

    def read_modelcount_results(self, no_trend: bool):
        """Read number of models contributing in the median with non-nan values"""
        if not no_trend:
            self.dominant_return_period_count = {}
            container = self.dominant_return_period_count
            output_subpath = "original"
        else:
            self.dominant_return_period_count_no_trend = {}
            container = self.dominant_return_period_count_no_trend
            output_subpath = "detrended"
        filenames = glob.glob(
            os.path.join(
                OUTPUT_PATH,
                output_subpath,
                "dominant_return_period",
                self.impact_type + "_" + self.impact_type if not RUN_CROP_TYPE_RESOLVED else self.crop_type,
            )
            + "/*.nc",
            root_dir=os.path.join(
                OUTPUT_PATH,
                output_subpath,
                "dominant_return_period",
                self.impact_type + "_" + self.impact_type if not RUN_CROP_TYPE_RESOLVED else self.crop_type,
            ),
        )
        for filename in filenames:
            if not filename.endswith(".nc"):
                self.log.error(
                    "File name extension is expected to be .nc "
                    f"but got {filename.split('.')[-1]} instead"
                )
                raise ValueError
            # get ssp and n_t from filename
            # remove filetype ending and split
            _filename = os.path.basename(filename).split(".")[0].split("_")
            # check if Nt and ssp match:
            if (
                _filename[-4] != "modelcounts"
                or int(_filename[-2].split("Nt")[1]) != NT
            ):
                continue
            if str(self.ssp_name) not in filename:
                continue
            self.log.info(f"Reading file {filename}...")
            # append data into a data set for each extreme event
            data = xr.open_dataset(filename)
            key = (self.impact_type, self.impact_type)
            container[key] = data
            self.log.info(f"{filename} successfully parsed!\n")
        self.log.info(
            "Parsing result model counts for dominant return period successful!"
        )

    # pylint: disable=too-many-branches
    def read_median_results(self, data_type: str, no_trend: bool):
        """Read result median dominant frequencies as netcdf"""
        if data_type == "dominant_return_period" and not no_trend:
            self.dominant_return_period_t0 = {}
            self.dominant_return_period_model = {}
            self.dominant_return_period_total = {}
            container_t0 = self.dominant_return_period_t0
            container_model = self.dominant_return_period_model
            container_total = self.dominant_return_period_total
            output_subpath = "original"
        elif data_type == "dominant_return_period" and no_trend:
            self.dominant_return_period_t0_no_trend = {}
            self.dominant_return_period_model_no_trend = {}
            self.dominant_return_period_total_no_trend = {}
            container_t0 = self.dominant_return_period_t0_no_trend
            container_model = self.dominant_return_period_model_no_trend
            container_total = self.dominant_return_period_total_no_trend
            output_subpath = "detrended"
        elif data_type == "event_counts":
            self.impact_count_t0 = {}
            self.impact_count_model = {}
            self.impact_count_total = {}
            container_t0 = self.impact_count_t0
            container_model = self.impact_count_model
            container_total = self.impact_count_total
            output_subpath = ""
        else:
            self.log.error(f"Result data type {data_type} is unknown!")
            raise ValueError
        filenames = (
            os.path.join(
                OUTPUT_PATH,
                output_subpath,
                data_type,
                self.impact_type + "_" + self.impact_type if not RUN_CROP_TYPE_RESOLVED else self.crop_type,
            )
            if data_type == "dominant_return_period"
            else os.path.join(
                OUTPUT_PATH,
                output_subpath,
                data_type,
                self.impact_type if not RUN_CROP_TYPE_RESOLVED else self.crop_type
            )
        )
        filenames = glob.glob(filenames + "/*.nc")
        for filename in filenames:
            if not filename.endswith(".nc"):
                self.log.error(
                    "File name extension is expected to be .nc "
                    f"but got {filename.split('.')[-1]} instead"
                )
                raise ValueError
            # get ssp and n_t from filename
            # remove filetype ending and split
            _filename = os.path.basename(filename).split(".")[0].split("_")
            # check if Nt and ssp match:
            if (
                _filename[-4] != "median"
                or int(_filename[-2].split("Nt")[1]) != NT
            ):
                continue
            # Falls self.ssp_name als String darin nicht vorkommt...
            if str(self.ssp_name) not in filename:
                continue
            self.log.info(f"Reading file {filename}...")
            # append data into a data set for each extreme event
            data = (
                xr.open_dataset(filename)
                if _filename[-1] != "total"
                else xr.Dataset({"total": xr.open_dataarray(filename)})
            )
            if data_type == "dominant_return_period":
                key = (self.impact_type, self.impact_type)
            else:
                key = self.impact_type
            if _filename[-1] == "t0":
                container_t0[key] = data
            elif _filename[-1] == "model":
                container_model[key] = data
            elif _filename[-1] == "total":
                container_total[key] = data
            self.log.info(f"{filename} successfully parsed!\n")
        self.log.info(f"Parsing result median {data_type} successful!")

    # pylint: disable=too-many-branches
    def read_std_results(self, data_type: str, no_trend: bool):
        """Read result median dominant frequencies as netcdf"""
        if data_type == "dominant_return_period" and not no_trend:
            self.dominant_return_period_t0_std = {}
            self.dominant_return_period_model_std = {}
            self.dominant_return_period_total_std = {}
            container_t0 = self.dominant_return_period_t0_std
            container_model = self.dominant_return_period_model_std
            container_total = self.dominant_return_period_total_std
            output_subpath = "original"
        elif data_type == "dominant_return_period" and no_trend:
            self.dominant_return_period_t0_no_trend_std = {}
            self.dominant_return_period_model_no_trend_std = {}
            self.dominant_return_period_total_no_trend_std = {}
            container_t0 = self.dominant_return_period_t0_no_trend_std
            container_model = self.dominant_return_period_model_no_trend_std
            container_total = self.dominant_return_period_total_no_trend_std
            output_subpath = "detrended"
        elif data_type == "event_counts":
            self.impact_count_t0_std = {}
            self.impact_count_model_std = {}
            self.impact_count_total_std = {}
            container_t0 = self.impact_count_t0_std
            container_model = self.impact_count_model_std
            container_total = self.impact_count_total_std
            output_subpath = ""
        else:
            self.log.error(f"Result data type {data_type} is unknown!")
            raise ValueError
        filenames = (
            os.path.join(
                OUTPUT_PATH,
                output_subpath,
                data_type,
                self.impact_type + "_" + self.impact_type if not RUN_CROP_TYPE_RESOLVED else self.crop_type,
            )
            if data_type == "dominant_return_period"
            else os.path.join(OUTPUT_PATH, output_subpath, data_type, self.impact_type)
        )
        filenames = glob.glob(filenames + "/*.nc")
        for filename in filenames:
            if not filename.endswith(".nc"):
                self.log.error(
                    "File name extension is expected to be .nc "
                    f"but got {filename.split('.')[-1]} instead"
                )
                raise ValueError
            # get ssp and n_t from filename
            # remove filetype ending and split
            _filename = os.path.basename(filename).split(".")[0].split("_")
            # check if Nt and ssp match:
            if (
                _filename[-4] != "std"
                or int(_filename[-2].split("Nt")[1]) != NT
            ):
                continue
            if str(self.ssp_name) not in filename:
                continue
            self.log.info(f"Reading file {filename}...")
            # append data into a data set for each extreme event
            data = (
                xr.open_dataset(filename)
                if _filename[-1] != "total"
                else xr.Dataset({"total": xr.open_dataarray(filename)})
            )
            if data_type == "dominant_return_period":
                key = (self.impact_type, self.impact_type)
            else:
                key = self.impact_type
            if _filename[-1] == "t0":
                container_t0[key] = data
            elif _filename[-1] == "model":
                container_model[key] = data
            elif _filename[-1] == "total":
                container_total[key] = data
            self.log.info(f"{filename} successfully parsed!\n")
        self.log.info(f"Parsing result median {data_type} successful!")

    def create_local_impact_plots(
        self,
    ) -> None:
        """Create LOCAL DYNAMIC data plots (LOCATIONS)"""
        # check time dependence of correlation close to location
        font = 25
        for location, (lat, long) in LOCATIONS.items():
            extreme_ds_local = self.impact_time_series.sel({"lon": long, "lat": lat})
            # make extreme event plot for Potsdam
            for event in extreme_ds_local.data_vars.keys():
                self.log.info(
                    f"Creating plots for {self.impact_type}-{event} extreme event in {location}..."
                )
                plt.figure(figsize=(6, 6))
                if self.impact_type == "cropfailedarea":
                    plt.ylabel("area fraction indicator", fontsize=font)
                plt.text(
                    0,
                    1,
                    ISIMIP_IMPACT_LABEL[self.impact_type],
                    transform=plt.gcf().transFigure,
                    horizontalalignment="left",
                    verticalalignment="top",
                    fontsize=font,
                    fontweight='bold'
                )
                plt.title(ISIMIP_IMPACT_NAME[self.impact_type], fontsize=font)
                plt.xlabel("year", fontsize=font)
                plt.xticks(fontsize=font, rotation=45)
                plt.yticks(fontsize=font)
                plt.scatter(
                    extreme_ds_local["time"].values,
                    extreme_ds_local[event].values,
                )
                # create vertical lines for relevant times and emphasize length of
                # correlation intervals by vertical line that are split along the
                # plotting window
                y_extent = (
                    extreme_ds_local[event].max().values
                    - extreme_ds_local[event].min().values
                )
                x_extent = max(t_0s) + 2 * NT - min(t_0s)
                plt.xlim(left=min(t_0s), right=max(t_0s) + 2 * NT)
                plt.ylim(bottom=0, top=1)
                if y_extent > 0:
                    plt.ylim(
                        bottom=extreme_ds_local[event].min() - 0.1 * y_extent,
                        top=extreme_ds_local[event].max() + 0.1 * y_extent,
                    )
                for i, t_0 in enumerate(t_0s):
                    plt.axvline(
                        x=t_0,
                        c="blue",
                        ymin=float(i) / len(t_0s),
                        ymax=(i + 1.0) / len(t_0s),
                        lw=2,
                    )
                    plt.axhline(
                        y=extreme_ds_local[event].min()
                        - 0.1 * y_extent
                        + (i + 0.5) * 1.2 * y_extent / len(t_0s),
                        c="blue",
                        xmin=(t_0 - min(t_0s)) / x_extent,
                        xmax=(t_0 + NT - min(t_0s)) / x_extent,
                        lw=2,
                    )
                    plt.axvline(
                        x=t_0 + NT,
                        c="red",
                        ymin=float(i) / len(t_0s),
                        ymax=(i + 1.0) / len(t_0s),
                        lw=2,
                    )
                    plt.axhline(
                        y=extreme_ds_local[event].min()
                        - 0.1 * y_extent
                        + (i + 0.5) * 1.2 * y_extent / len(t_0s),
                        c="red",
                        xmin=(t_0 + NT - min(t_0s)) / x_extent,
                        xmax=(t_0 + 2 * NT - min(t_0s)) / x_extent,
                        lw=2,
                    )
                    plt.axvline(
                        x=t_0 + 2 * NT,
                        c="red",
                        ymin=float(i) / len(t_0s),
                        ymax=(i + 1.0) / len(t_0s),
                        lw=2,
                    )
                full_path = os.path.join(
                    os.path.join(
                        PLOTS_PATH,
                        f"Nt_{NT}",
                        f"{self.ssp_name}",
                        "local_dynamic_plots",
                        f"{location}",
                    )
                )
                if not os.path.exists(full_path):
                    os.makedirs(full_path)
                plt.tight_layout()
                plt.savefig(
                    os.path.join(
                        full_path,
                        f"{location}_{self.ssp_name}_{self.impact_type}_{event.split('_')[1]}_"
                        f"{event.split('_')[0]}_vs_time.pdf",
                    ),
                    format='pdf',
                )
                plt.cla()
                plt.clf()
                plt.close()

    # pylint: disable=too-many-locals, too-many-branches
    def create_local_correlation_plots(self, no_trend: bool) -> None:
        """Create LOCAL DYNAMIC plots (LOCATIONS)"""
        font = 25
        line_width = 4
        point_size = 150
        if not no_trend:
            container = self.local_dominant_return_period
            trend_particle = ""
        else:
            container = self.local_dominant_return_period_no_trend
            trend_particle = "detrended"
        # check time dependence of correlation close to location
        # pylint: disable=too-many-locals, too-many-nested-blocks
        for location, local_data in container.items():
            # make extreme event plot for Potsdam
            for event, specific_data in local_data.items():
                # make sure that the climate models are the same
                self.log.info(
                    f"Creating correlation plots for {self.impact_type}-{event} "
                    f"extreme event in {location}..."
                )
                # correlation function is evaluated for a time window of Nt
                for t_0, data in specific_data.items():
                    # determine limits of time window, namely [t,t+Nt)
                    # for f_i and [t+n,t+Nt+n) for f_j, where n=0,..,Nt-1
                    t_start = t_0
                    plt.figure(figsize=(6, 6))
                    plt.axes().yaxis.get_offset_text().set_fontsize(2 * font / 3)
                    if self.impact_type == "cropfailedarea" or NT == 125:
                        plt.ylabel("time correlation", fontsize=font)
                    if NT != 125:
                        plt.text(
                            0,
                            1,
                            ISIMIP_IMPACT_LABEL[self.impact_type],
                            transform=plt.gcf().transFigure,
                            horizontalalignment="left",
                            verticalalignment="top",
                            fontsize=font,
                            fontweight='bold'
                        )
                    if USE_AR_MODEL and (data["dominant_ret_per"] is not None or data.get('AR dominant period') is not None):
                        plt.title(
                            f"{ISIMIP_IMPACT_NAME[self.impact_type]}\n "
                            f"dominant per. (AR)= {data['dominant_ret_per']}y ({data.get('AR dominant period')}y)",
                            fontsize=font,
                        )
                    elif data["dominant_ret_per"] is not None:
                        plt.title(
                            f"{ISIMIP_IMPACT_NAME[self.impact_type]}\n "
                            f"dominant per. = {data['dominant_ret_per']}y",
                            fontsize=font,
                        )
                    else:
                        plt.title(
                            f"{ISIMIP_IMPACT_NAME[self.impact_type]}\n", fontsize=font
                        )
                    plt.xlabel("time lag [y]", fontsize=font)
                    plt.xticks(fontsize=font)
                    plt.yticks(fontsize=font)
                    plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
                    # plot correlation function
                    plt.scatter(
                        list(range(NT)),
                        data["corr"],
                        label="data",
                        s=point_size,
                    )
                    # plot Fourier fit for the first n largest coefficients
                    # initialize Fourier coefficients with zero
                    fit_n = np.array([0.0 + 1j * 0.0] * len(data["c_n"]))
                    # add next-largest coefficient one by one
                    for n_idx, idx in enumerate(data["sorted_idx"][:4]):
                        # replace zero by coefficient at proper position
                        fit_n[idx] = data["c_n"][idx]
                        # do inverse Fourier transformation to get fit
                        fourier_fit = np.fft.irfft(fit_n, len(data["corr"]))
                        plt.plot(
                            list(range(NT)),
                            fourier_fit,
                            label=f"{n_idx+1} coeff. fit",
                            marker=FOURIER_MARKERSTYLE[n_idx + 1],
                            markersize=8,
                            lw=line_width,
                        )
                    plt.plot(
                        list(range(NT)),
                        data["max_fit"],
                        label="Fourier fit",
                        lw=line_width,
                    )
                    if self.impact_type == "heatwavedarea" or NT == 125:
                        plt.legend(
                            ncol=2,
                            fancybox=True,
                            shadow=True,
                            prop={"size": 3 * font / 4},
                            handlelength=1,
                        )
                    full_path = os.path.join(
                        os.path.join(
                            PLOTS_PATH,
                            f"Nt_{NT}",
                            f"{self.ssp_name}",
                            "local_dynamic_plots",
                            f"{location}",
                        )
                    )
                    if not os.path.exists(full_path):
                        os.makedirs(full_path)
                    plt.tight_layout()
                    plt.savefig(
                        os.path.join(
                            full_path,
                            f"{t_start}_{self.ssp_name}"
                            f"_{self.impact_type}_{event[0].split('_')[1]}_{event[0].split('_')[0]}"
                            f"_{self.impact_type}_{event[1].split('_')[1]}_{event[1].split('_')[0]}"
                            f"_{location}_time_correlation" + trend_particle + ".pdf",
                        ),
                        format='pdf',
                    )
                    plt.cla()
                    plt.clf()
                    plt.close()

    def create_global_plots(
        self,
    ) -> None:
        """create GLOBAL plots"""
        # create global extreme event plot for start time
        extreme_ds_t0 = self.impact_time_series.sel(time=min(t_0s))
        # make extreme event plot for reference time
        for event in extreme_ds_t0.data_vars.keys():
            self.log.info(
                f"Creating fixed time global plot for {self.impact_type}-{event} extreme event..."
            )
            fig, axis = plt.subplots(figsize=(48, 38))
            title = (
                f"{self.impact_type}_{event.split('_')[1]} {event.split('_')[0]} "
                f"{self.ssp_name} {min(t_0s)}"
                if not self.use_model_mean
                else f"{self.impact_type}-{event.split('_')[1]} model mean "
                f"{self.ssp_name} {min(t_0s)}"
            )
            plt.title(title)
            plt.xlabel("longitude [°]")
            plt.ylabel("latitude [°]")
            xr.plot.pcolormesh(
                darray=extreme_ds_t0[event],
                cmap="YlOrRd",
                ax=axis,
                rasterized=True
            )
            world = gpd.read_file(
                os.path.abspath(
                    os.path.join(os.path.dirname(__file__), "..", "data", "ne_110m_admin_0_countries.zip")
                )
            )
            world.boundary.plot(color="gray", ax=axis, linewidth=1)
            full_path = os.path.join(
                os.path.join(
                    PLOTS_PATH, f"Nt_{NT}", f"{self.ssp_name}", "global_static_plots"
                )
            )
            if not os.path.exists(full_path):
                os.makedirs(full_path)
            fig.savefig(
                os.path.join(
                    full_path,
                    f"{self.impact_type}_{event.split('_')[1]}_{event.split('_')[0]}"
                    "_vs_position_t0.pdf",
                ),
                bbox_inches="tight",
                format='pdf',
            )
            plt.cla()
            plt.clf()
            plt.close()
            self.log.info("Plot successfully created!\n")

    # pylint: disable=too-many-locals, too-many-statements
    def create_separate_correlation_plots(
        self, no_trend: bool, aggregation: str = "t0"
    ) -> None:
        """create global correlation plots"""
        # decide for aggregation level
        trend_particle = "" if not no_trend else "detrended"
        if aggregation == "t_0":
            container = (
                self.dominant_return_period_t0
                if not no_trend
                else self.dominant_return_period_t0_no_trend
            )
            impact_counts = self.impact_count_t0
        elif aggregation == "model":
            container = (
                self.dominant_return_period_model
                if not no_trend
                else self.dominant_return_period_model_no_trend
            )
            impact_counts = self.impact_count_model
        elif aggregation == "total":
            container = (
                self.dominant_return_period_total
                if not no_trend
                else self.dominant_return_period_total_no_trend
            )
            impact_counts = self.impact_count_total
        # make global correlation time plot
        # correlation function is evaluated for a time window of Nt
        for (event, event1), data in container.items():
            cmap = cmc.batlow_r # plt.colormaps["turbo_r"]
            cmaplist = [cmap(i) for i in range(cmap.N)]
            # force the first color entry (None values) to be grey
            cmaplist[0] = (0.8, 0.8, 0.8, 1.0)
            # create the new map
            cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
                "Custom cmap", cmaplist, cmap.N
            )
            # define color bins according to return periods
            d_bin = 3
            adjusted_NT = NT if NT >= 25 else 25
            bins = np.sort(np.append(-2, np.arange(1, adjusted_NT + d_bin, d_bin)))
            # define discrete colormap
            norm = matplotlib.colors.BoundaryNorm(bins, cmap.N)
            # great world map as background
            # world = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))
            # create plot for each data set variable
            for var in data.data_vars:
                if aggregation == "t_0":
                    titel_str = f"[{int(var)}, {int(var) + NT * 2})"
                else:
                    titel_str = var
                # determine limits of time window
                self.log.info(
                    f"Creating plot for {event}-{event1} for {titel_str} "
                    "and global time correlation..."
                )
                # replace non-trivial nan-values (irregular impacts) with zero value
                dominant_freq = xr.where(
                    np.logical_or(impact_counts[event][var] == 0, ~np.isnan(data[var])),
                    data[var],
                    0,
                )
                irregular_area_share = None
                # read crop land area if crop failure
                if self.impact_type == "cropfailedarea":
                    landuse = xr.open_dataset(
                        os.path.join(
                            OUTPUT_PATH, "landuse", "landuse_cropland_total_2015.nc"
                        ),
                        decode_times=False,
                    )
                    grid_area = xr.open_dataset(SURFACE_AREA_PATH)
                    irregular_area_share = (
                        xr.where(dominant_freq == 0, 1, 0)
                        * landuse["cropland_total"]
                        * grid_area
                    ).sum() / (
                        xr.where(dominant_freq.notnull(), 1, 0)
                        * landuse["cropland_total"]
                        * grid_area
                    ).sum()
                # create max time plot
                font = 10
                plt.rcParams.update({"font.size": font})
                fig, axis = plt.subplots(
                    figsize=(6, 3), subplot_kw={"projection": ccrs.Robinson()}
                )
                axis.coastlines(color="grey")
                if self.use_result_median:
                    gcm_model = "median"
                    impact_model = "median"
                    impact_type = event
                else:
                    gcm_model = event.split("_")[1]
                    impact_type = self.impact_type
                    impact_model = event.split("_")[0]
                xr.plot.pcolormesh(
                    darray=dominant_freq,
                    cmap=cmap,
                    norm=norm,
                    transform=ccrs.PlateCarree(),
                    ax=axis,
                    add_colorbar=False,
                    rasterized=True
                )
                if aggregation == "t_0" and self.use_result_median and self.ssp_name == "ssp585" and not no_trend:
                    plt.title(f"{int(var)} - {int(var)+2*NT-1}")
                elif aggregation == "t_0" and self.use_result_median and self.ssp_name == "ssp585" and no_trend:
                    plt.title(f"{int(var)} - {int(var) + 2 * NT - 1} (detrended)")
                else:
                    plt.title("")
                axis.set_global()
                # remove frame and ticks
                axis.spines["top"].set_visible(False)
                axis.spines["right"].set_visible(False)
                axis.spines["bottom"].set_visible(False)
                axis.spines["left"].set_visible(False)
                axis.get_xaxis().set_ticks([])
                axis.get_yaxis().set_ticks([])
                wildfire_cap = 0 if self.wildfire_cap is None else self.wildfire_cap
                # set figure title
                if RUN_CROP_TYPE_RESOLVED:
                    fig_label = FIG_LABEL_SINGLECROP[self.crop_type]
                    axis.set_title(self.crop_type.capitalize())
                elif (
                    (np.any(np.isclose(wildfire_cap, [0.5, 20, 1])) and gcm_model != "gfdl-esm4")
                    or (
                        self.ssp_name in ["ssp126", "ssp370"]
                        and self.impact_type in ["cropfailedarea", "heatwavedarea"]
                        and not no_trend
                    )
                    or (
                        self.impact_type == "burntarea"
                        and self.ssp_name == "ssp126"
                    )
                    or gcm_model == "gfdl-esm4" and impact_model == "classic"
                ):
                    fig_label = "a"
                elif (
                    np.any(np.isclose(wildfire_cap, [50, 10, 1.5]))
                    or (
                        self.ssp_name in ["ssp126", "ssp370"]
                        and self.impact_type in ["cropfailedarea", "heatwavedarea"]
                        and no_trend
                    )
                    or (
                        self.impact_type == "burntarea"
                        and self.ssp_name == "ssp370"
                    )
                    or gcm_model == "gfdl-esm4" and impact_model == "lpjml5-7-10-fire"
                    or var == "2040" and not no_trend
                ):
                    fig_label = "b"
                elif (
                    np.any(np.isclose(wildfire_cap, [100, 5, 2]))
                    or gcm_model == "gfdl-esm4" and impact_model == "visit"
                    or var == "2040" and no_trend
                ):
                    fig_label = "c"
                elif (
                    not self.is_r2test and
                    np.isclose(R2_THRESHOLD, 0.5)
                    and self.ssp_name == "picontrol"
                    or self.ssp_name == "historical"
                ):
                    fig_label = ISIMIP_IMPACT_LABEL[self.impact_type]
                    axis.set_title(ISIMIP_IMPACT_NAME[self.impact_type])
                elif (
                    np.isclose(R2_THRESHOLD, 0.5)
                    and self.ssp_name == "picontrol"
                    and self.is_r2test
                ):
                    fig_label = "b"
                elif np.isclose(R2_THRESHOLD, 0.4) and self.ssp_name == "picontrol":
                    fig_label = "a"
                elif self.ssp_name == "picontrol":
                    fig_label = "c"
                elif self.ssp_name == "ssp585" and var == "1950":
                    fig_label = "a"
                elif self.ssp_name == "ssp585" and var in ["2050", "2070"] and not no_trend:
                    fig_label = "a"
                elif self.ssp_name == "ssp585" and var in ["2050", "2070"] and no_trend:
                    fig_label = "b"
                else:
                    fig_label = "c"
                plt.text(
                    0.09,
                    0.9,
                    fig_label,
                    transform=plt.gcf().transFigure,
                    horizontalalignment="left",
                    verticalalignment="top",
                    fontsize=2*font,
                    fontweight='bold'
                )
                # world.boundary.plot(color="gray", ax=axis, linewidth=1)

                # create two axes for the colorbar (None and non-None)
                axis2 = fig.add_axes([0.9, 0.15, 0.03, 0.7])
                axis3 = fig.add_axes([0.9, 0.08, 0.03, 0.7 / (len(bins) - 2)])
                # non-None color bar
                cmaplist = [cmap(i) for i in range(cmap.N)]
                # force the first color entry (None values) to be white
                cmaplist[0] = (0, 0, 0, 0.0)
                # create the new map
                cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
                    "Custom cmap", cmaplist, cmap.N
                )
                cbar2 = matplotlib.colorbar.ColorbarBase(
                    axis2,
                    cmap=cmap,
                    norm=norm,
                    spacing="uniform",
                    ticks=bins[1:],
                    boundaries=bins[1:],
                    format="%.1f",
                )
                cbar2.set_ticks(ticks=bins[1:], labels=bins[1:])
                cbar2.ax.set_ylabel("Dominant period [year]", rotation=90, labelpad=20)
                # None colorbar
                cmaplist = [cmap(i) for i in range(cmap.N)]
                # force the first color entry (None values) to be grey
                cmaplist[0] = (0.8, 0.8, 0.8, 1.0)
                # create the new map
                cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
                    "Custom cmap", cmaplist, cmap.N
                )
                cbar3 = matplotlib.colorbar.ColorbarBase(
                    axis3,
                    cmap=cmap,
                    spacing="uniform",
                    ticks=[bins[0]],
                    boundaries=bins[:2],
                    format="%.1f",
                )
                cbar3.set_ticks(ticks=[(bins[0] + bins[1]) / 2], labels=["None"])
                # inset histogram plot
                left, bottom, width, height = [0.17, 0.15, 0.15, 0.2]
                inset_ax = fig.add_axes([left, bottom, width, height])
                bin_colors = ScalarMappable(norm=norm, cmap=cmap).to_rgba(bins[:-1])
                data_plt = dominant_freq.to_dataframe(name="val").reset_index(level=["lon", "lat"])
                *_, bars = inset_ax.hist(
                    data_plt["val"].dropna(),
                    bins=bins,
                    weights=np.ones_like(data_plt["val"].dropna())
                    / len(data_plt["val"].dropna()),
                )
                for i, bar_i in enumerate(bars):
                    bar_i.set_facecolor(bin_colors[i])
                # inset_ax.set_ylim([0, max(bar_i.get_height() for bar_i in bars)])
                inset_ax.set_ylim([0, 1])
                inset_ax.set_xticks([])
                inset_ax.set_title("total distribution")
                inset_ax.set_xlabel("Dominant period")
                # store figure
                full_path = os.path.join(
                    PLOTS_PATH, f"Nt_{NT}", f"{self.ssp_name}", "global_dynamic_plots"
                )
                if not os.path.exists(full_path):
                    os.makedirs(full_path)
                impact_particle = impact_type if not RUN_CROP_TYPE_RESOLVED else self.crop_type
                fig.savefig(
                    os.path.join(
                        full_path,
                        f"{var}_{self.ssp_name}_"
                        f"{impact_particle}_{gcm_model}_{impact_model}_"
                        f"{impact_particle}_{gcm_model}_{impact_model}_frequency_max"
                        + trend_particle
                        + ".pdf",
                    ),
                    bbox_inches="tight",
                    format='pdf',
                )
                # clear axis to add value plot
                plt.cla()
                plt.clf()
                plt.close()
                # write total distribution to csv file
                full_path = "detrended" if no_trend else "original"
                full_path = os.path.join(
                    OUTPUT_PATH,
                    full_path,
                    "dominant_return_period",
                    f"{impact_type}_{impact_type}",
                    "total_distribution",
                )
                if not os.path.exists(full_path):
                    os.makedirs(full_path)
                impact_particle = impact_type if not RUN_CROP_TYPE_RESOLVED else self.crop_type
                filename = os.path.join(
                    full_path,
                    f"{var}_NT{NT}_{self.ssp_name}_"
                    f"{impact_particle}_{gcm_model}_{impact_model}_"
                    f"{impact_particle}_{gcm_model}_{impact_model}_total_distribution_croparea"
                    + trend_particle
                    + ".csv",
                )
                if self.impact_type == "cropfailedarea":
                    with open(filename, "w", encoding="utf-8") as csvfile:
                        csvwriter = csv.writer(csvfile)
                        csvwriter.writerows(
                            ["irregular", irregular_area_share.to_array().values]
                        )
                filename = filename.replace("croparea", "gridcell")
                with open(filename, "w", encoding="utf-8") as csvfile:
                    csvwriter = csv.writer(csvfile)
                    if not np.isclose(
                        np.sum([bar_i.get_height() for bar_i in bars]), 1, rtol=1e-6
                    ):
                        raise ValueError(
                            "sum of histogram does not add up to one: "
                            f"{np.sum([bar_i.get_height() for bar_i in bars])}"
                        )
                    csvwriter.writerows(
                        list(
                            zip(
                                np.where(bins != -2, bins, "None"),
                                [bar_i.get_height() for bar_i in bars],
                            )
                        )
                    )
                self.log.info("Plot finished!\n")

    # pylint: disable=too-many-locals, too-many-statements
    def create_separate_correlation_plots_std(
        self, no_trend: bool, aggregation: str = "t0"
    ) -> None:
        """create global correlation plots for std instead of median"""
        # decide for aggregation level
        trend_particle = "" if not no_trend else "detrended"
        if aggregation == "t_0":
            container = (
                self.dominant_return_period_t0_std
                if not no_trend
                else self.dominant_return_period_t0_no_trend_std
            )
            impact_counts = self.impact_count_t0
        elif aggregation == "model":
            container = (
                self.dominant_return_period_model_std
                if not no_trend
                else self.dominant_return_period_model_no_trend_std
            )
            impact_counts = self.impact_count_model
        elif aggregation == "total":
            container = (
                self.dominant_return_period_total_std
                if not no_trend
                else self.dominant_return_period_total_no_trend_std
            )
            impact_counts = self.impact_count_total
        # make global correlation time plot
        # correlation function is evaluated for a time window of Nt
        for (event, event1), data in container.items():
            cmap = cmc.batlow_r  # plt.colormaps["turbo_r"]
            cmaplist = [cmap(i) for i in range(cmap.N)]
            # force the first color entry (None values) to be grey
            cmaplist[0] = (0.8, 0.8, 0.8, 1.0)
            # create the new map
            cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
                "Custom cmap", cmaplist, cmap.N
            )
            # define color bins according to return periods
            d_bin = 3
            adjusted_NT = NT if NT >= 25 else 25
            bins = np.sort(np.append(-2, np.arange(1, adjusted_NT + d_bin, d_bin)))
            # define discrete colormap
            norm = matplotlib.colors.BoundaryNorm(bins, cmap.N)
            # create plot for each data set variable
            for var in data.data_vars:
                if aggregation == "t_0":
                    titel_str = f"[{int(var)}, {int(var) + NT * 2})"
                else:
                    titel_str = var
                # determine limits of time window
                self.log.info(
                    f"Creating plot for {event}-{event1} for {titel_str} "
                    "and global time correlation..."
                )
                # replace non-trivial nan-values (irregular impacts) with zero value
                dominant_freq = xr.where(
                    np.logical_or(impact_counts[event][var] == 0, ~np.isnan(data[var])),
                    data[var],
                    -1,
                )
                irregular_area_share = None
                # read crop land area if crop failure
                if self.impact_type == "cropfailedarea":
                    landuse = xr.open_dataset(
                        os.path.join(
                            OUTPUT_PATH, "landuse", "landuse_cropland_total_2015.nc"
                        ),
                        decode_times=False,
                    )
                    grid_area = xr.open_dataset(SURFACE_AREA_PATH)
                    irregular_area_share = (
                        xr.where(dominant_freq == -1, 1, 0)
                        * landuse["cropland_total"]
                        * grid_area
                    ).sum() / (
                        xr.where(dominant_freq.notnull(), 1, 0)
                        * landuse["cropland_total"]
                        * grid_area
                    ).sum()
                # create max time plot
                font = 10
                plt.rcParams.update({"font.size": font})
                fig, axis = plt.subplots(
                    figsize=(6, 3), subplot_kw={"projection": ccrs.Robinson()}
                )
                axis.coastlines(color="grey")
                if self.use_result_median:
                    gcm_model = "std"
                    impact_model = "std"
                    impact_type = event
                else:
                    gcm_model = event.split("_")[1]
                    impact_type = self.impact_type
                    impact_model = event.split("_")[0]
                xr.plot.pcolormesh(
                    darray=dominant_freq,
                    cmap=cmap,
                    norm=norm,
                    transform=ccrs.PlateCarree(),
                    ax=axis,
                    add_colorbar=False,
                    rasterized=True,
                )
                axis.set_global()
                # add cbar
                axis2 = fig.add_axes([0.9, 0.15, 0.03, 0.7])
                cbar2 = matplotlib.colorbar.ColorbarBase(
                    axis2,
                    cmap=cmap,
                    norm=norm,
                    spacing="uniform",
                    ticks=bins,
                    boundaries=bins,
                    format="%.1f",
                )
                cbar2.set_ticks(ticks=bins, labels=bins)
                cbar2.ax.set_ylabel(
                    "Number of extreme events", rotation=90, labelpad=20
                )
                # remove frame and ticks
                axis.spines["top"].set_visible(False)
                axis.spines["right"].set_visible(False)
                axis.spines["bottom"].set_visible(False)
                axis.spines["left"].set_visible(False)
                axis.get_xaxis().set_ticks([])
                axis.get_yaxis().set_ticks([])
                # set figure title
                if self.impact_type == "cropfailedarea":
                    fig_label = "a"
                elif self.impact_type == "heatwavedarea":
                    fig_label = "b"
                else:
                    fig_label = "c"
                plt.text(
                    0.09,
                    0.9,
                    fig_label,
                    transform=plt.gcf().transFigure,
                    horizontalalignment="left",
                    verticalalignment="top",
                    fontsize=2*font,
                    fontweight='bold'
                )

                # create two axes for the colorbar (None and non-None)
                axis2 = fig.add_axes([0.9, 0.15, 0.03, 0.7])
                axis3 = fig.add_axes([0.9, 0.08, 0.03, 0.7 / (len(bins) - 2)])
                # non-None color bar
                cmaplist = [cmap(i) for i in range(cmap.N)]
                # force the first color entry (None values) to be white
                cmaplist[0] = (0, 0, 0, 0.0)
                # create the new map
                cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
                    "Custom cmap", cmaplist, cmap.N
                )
                cbar2 = matplotlib.colorbar.ColorbarBase(
                    axis2,
                    cmap=cmap,
                    norm=norm,
                    spacing="uniform",
                    ticks=bins,
                    boundaries=bins,
                    format="%.1f",
                )
                cbar2.set_ticks(ticks=bins, labels=bins)
                cbar2.ax.set_ylabel(
                    "Number of extreme events", rotation=90, labelpad=20
                )
                # None colorbar
                cmaplist = [cmap(i) for i in range(cmap.N)]
                # force the first color entry (None values) to be grey
                cmaplist[0] = (0.8, 0.8, 0.8, 1.0)
                # create the new map
                cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
                    "Custom cmap", cmaplist, cmap.N
                )
                cbar3 = matplotlib.colorbar.ColorbarBase(
                    axis3,
                    cmap=cmap,
                    spacing="uniform",
                    ticks=[bins[0]],
                    boundaries=bins[:2],
                    format="%.1f",
                )
                cbar3.set_ticks(ticks=[(bins[0] + bins[1]) / 2], labels=["None"])
                # inset histogram plot
                left, bottom, width, height = [0.17, 0.15, 0.15, 0.2]
                inset_ax = fig.add_axes([left, bottom, width, height])
                bin_colors = ScalarMappable(norm=norm, cmap=cmap).to_rgba(bins[:-1])
                data_plt = dominant_freq.to_dataframe(name="val").reset_index(level=["lon", "lat"])
                *_, bars = inset_ax.hist(
                    data_plt["val"].dropna(),
                    bins=bins,
                    weights=np.ones_like(data_plt["val"].dropna())
                    / len(data_plt["val"].dropna()),
                )
                for i, bar_i in enumerate(bars):
                    bar_i.set_facecolor(bin_colors[i])
                inset_ax.set_ylim([0, max(bar_i.get_height() for bar_i in bars)])
                inset_ax.set_xticks([])
                inset_ax.set_title("total distribution")
                inset_ax.set_xlabel("Dominant period")
                # store figure
                full_path = os.path.join(
                    PLOTS_PATH, f"Nt_{NT}", f"{self.ssp_name}", "global_dynamic_plots"
                )
                if not os.path.exists(full_path):
                    os.makedirs(full_path)
                impact_particle = impact_type if not RUN_CROP_TYPE_RESOLVED else self.crop_type
                fig.savefig(
                    os.path.join(
                        full_path,
                        f"{var}_{self.ssp_name}_"
                        f"{impact_particle}_{gcm_model}_{impact_model}_"
                        f"{impact_particle}_{gcm_model}_{impact_model}_dominant_period_std"
                        + trend_particle
                        + ".pdf",
                    ),
                    bbox_inches="tight",
                    format='pdf',
                )
                # clear axis to add value plot
                plt.cla()
                plt.clf()
                plt.close()
                # write total distribution to csv file
                full_path = "detrended" if no_trend else "original"
                full_path = os.path.join(
                    OUTPUT_PATH,
                    full_path,
                    "dominant_return_period",
                    f"{impact_type}_{impact_type}",
                    "total_distribution",
                )
                if not os.path.exists(full_path):
                    os.makedirs(full_path)
                impact_particle = impact_type if not RUN_CROP_TYPE_RESOLVED else self.crop_type
                filename = os.path.join(
                    full_path,
                    f"{var}_NT{NT}_{self.ssp_name}_"
                    f"{impact_particle}_{gcm_model}_{impact_model}_"
                    f"{impact_particle}_{gcm_model}_{impact_model}_total_distribution_std_croparea"
                    + trend_particle
                    + ".csv",
                )
                if self.impact_type == "cropfailedarea":
                    with open(filename, "w", encoding="utf-8") as csvfile:
                        csvwriter = csv.writer(csvfile)
                        csvwriter.writerows(
                            ["irregular", irregular_area_share.to_array().values]
                        )
                filename = filename.replace("croparea", "gridcell")
                with open(filename, "w", encoding="utf-8") as csvfile:
                    csvwriter = csv.writer(csvfile)
                    if not np.isclose(
                        np.sum([bar_i.get_height() for bar_i in bars]), 1, rtol=1e-6
                    ):
                        raise ValueError(
                            "sum of histogram does not add up to one: "
                            f"{np.sum([bar_i.get_height() for bar_i in bars])}"
                        )
                    csvwriter.writerows(
                        list(
                            zip(
                                np.where(bins != -2, bins, "None"),
                                [bar_i.get_height() for bar_i in bars],
                            )
                        )
                    )
                self.log.info("Plot finished!\n")

    def create_counting_time_plots(self) -> None:
        """This function creates plots for the impact count time series"""
        for event, data in self.total_count.items():
            impact_model = (
                event.split("_")[0] if not self.use_result_median else "result_median"
            )
            self.log.info(
                f"Creating extreme event total counting plot for {self.impact_type}-{event}..."
            )
            fontsize = 17
            plt.rcParams.update({"font.size": fontsize})
            fig, _ = plt.subplots(figsize=(6, 3))
            plt.xlabel("year")
            plt.xticks(fontsize=fontsize)
            plt.ylabel("total affected area\n[km$^2$]")
            plt.yticks(fontsize=fontsize)
            # add GCM curves
            if self.use_result_median:
                for i_mod, ((impact, gcm), data_gcm) in enumerate(
                    self.total_model_count_median.items()
                ):
                    if impact == event:
                        plt.errorbar(
                            data_gcm["year"].values,
                            data_gcm["counts"].values,
                            yerr=np.array(
                                [
                                    data_gcm["counts"].values
                                    - self.total_model_count_min[(impact, gcm)][
                                        "counts"
                                    ].values,
                                    abs(
                                        data_gcm["counts"].values
                                        - self.total_model_count_max[(impact, gcm)][
                                            "counts"
                                        ].values
                                    ),
                                ]
                            ),
                            errorevery=(
                                1 + i_mod,
                                2 * len(self.total_model_count_median),
                            ),
                            capsize=5,
                            capthick=2.5,
                            elinewidth=2.5,
                            linestyle="-",
                            c=ISMIP_GCM_COLOR[gcm],
                            label=gcm.upper(),
                            lw=4,
                        )
            else:
                plt.plot(
                    data["year"].values,
                    data["counts"].values,
                    ".-",
                    label="total median",
                    lw=30,
                )
            plt.title(
                f"{ISIMIP_IMPACT_NAME[self.impact_type]}, {self.ssp_name}", y=1.05, fontsize=fontsize
            )
            plt.text(
                0,
                1.05,
                ISIMIP_IMPACT_LABEL[self.impact_type] if self.ssp_name != "picontrol" else ISIMIP_IMPACT_PICONTROL_LABEL[self.impact_type],
                transform=plt.gcf().transFigure,
                horizontalalignment="left",
                verticalalignment="top",
                fontsize=2*fontsize,
                fontweight='bold'
            )
            # add legend only in cropfailedarea plot
            if event == "cropfailedarea" and self.ssp_name != "picontrol":
                plt.legend()
            full_path = os.path.join(
                PLOTS_PATH, f"Nt_{NT}", f"{self.ssp_name}", "global_counting_plots"
            )
            if not os.path.exists(full_path):
                os.makedirs(full_path)
            fig.savefig(
                os.path.join(
                    full_path,
                    f"total_{self.ssp_name}_{self.impact_type}_{impact_model}_"
                    "total_event_counts.pdf",
                ),
                bbox_inches="tight",
                format='pdf',
            )
            plt.tight_layout()
            plt.cla()
            plt.clf()
            plt.close()
            self.log.info("Plot successfully created!\n")

    def create_impact_count_plots(self) -> None:
        """This function creates plots for the impact counts"""
        cmap = cmc.batlow  # plt.colormaps["turbo"]
        # define color bins according to return periods
        d_bin = 5
        bins = np.sort(np.arange(0, 2 * NT + d_bin, d_bin))
        # define discrete colormap
        norm = matplotlib.colors.BoundaryNorm(bins, cmap.N)
        # make extreme event count plot for each reference time and event
        for event, item in self.impact_count_t0.items():
            impact_model = (
                event.split("_")[0] if not self.use_result_median else "result_median"
            )
            gcm_model = (
                event.split("_")[1] if not self.use_result_median else "result_median"
            )
            for reference_time in item.keys():
                self.log.info(
                    f"Creating extreme event counting plot for {event, reference_time}..."
                )
                font_size = 10
                plt.rcParams.update({"font.size": font_size})
                fig, axis = plt.subplots(
                    figsize=(6, 3), subplot_kw={"projection": ccrs.Robinson()}
                )
                axis.coastlines(color="grey")
                xr.plot.pcolormesh(
                    item.where(item != 0)[reference_time],
                    cmap=cmap,
                    norm=norm,
                    ax=axis,
                    transform=ccrs.PlateCarree(),
                    add_colorbar=False,
                    rasterized=True
                )
                axis.set_global()
                # add cbar
                axis2 = fig.add_axes([0.9, 0.15, 0.03, 0.7])
                cbar2 = matplotlib.colorbar.ColorbarBase(
                    axis2,
                    cmap=cmap,
                    norm=norm,
                    spacing="uniform",
                    ticks=bins,
                    boundaries=bins,
                    format="%.1f",
                )
                cbar2.set_ticks(ticks=bins, labels=bins)
                cbar2.ax.set_ylabel(
                    "Number of extreme events", rotation=90, labelpad=20
                )
                # remove frame and ticks
                axis.spines["top"].set_visible(False)
                axis.spines["right"].set_visible(False)
                axis.spines["bottom"].set_visible(False)
                axis.spines["left"].set_visible(False)
                axis.get_xaxis().set_ticks([])
                axis.get_yaxis().set_ticks([])
                axis.set_title(
                    f"{ISIMIP_IMPACT_NAME[self.impact_type]}, "
                    f"{int(reference_time)} - {int(reference_time)+2*NT-1}"
                )
                if int(reference_time) == 1850:
                    figure_label = "a"
                elif int(reference_time) == 1950:
                    figure_label = "b"
                else:
                    figure_label = "c"
                plt.text(
                    0.09,
                    0.9,
                    figure_label,
                    transform=plt.gcf().transFigure,
                    horizontalalignment="left",
                    verticalalignment="top",
                    fontsize=2*font_size,
                    fontweight='bold'
                )
                full_path = os.path.join(
                    PLOTS_PATH, f"Nt_{NT}", f"{self.ssp_name}", "global_counting_plots"
                )
                if not os.path.exists(full_path):
                    os.makedirs(full_path)
                impact_particle = self.impact_type if not RUN_CROP_TYPE_RESOLVED else self.crop_type
                fig.savefig(
                    os.path.join(
                        full_path,
                        f"{impact_particle}_{gcm_model}_{impact_model}_{reference_time}"
                        "_vs_position_event_counts.pdf",
                    ),
                    bbox_inches="tight",
                    format='pdf',
                )
                plt.cla()
                plt.clf()
                plt.close()
                self.log.info("Plot successfully created!\n")

    def create_model_count_plots(self, no_trend: bool):
        """Create plot of the number of models with non-zero dominant return period"""
        container = (
            self.dominant_return_period_count
            if not no_trend
            else self.dominant_return_period_count_no_trend
        ) if self.use_result_median else (
            {(self.impact_type, self.impact_type): xr.concat(list(self.dominant_return_period_t0.values()), dim="model")}
            if not no_trend
            else {(self.impact_type, self.impact_type): xr.concat(list(self.dominant_return_period_t0_no_trend.values()), dim="model")}
        )
        # correlation function is evaluated for a time window of Nt
        for (event, event1), data in container.items():
            # create plot for each data set variable
            cmap = cmc.batlow  # plt.colormaps["turbo"]
            # define color bins according to return periods
            d_bin = 1 if event != "cropfailedarea" else len(ALL_GCM_MODELS[event])
            bins = np.sort(
                np.arange(
                    0,
                    len(ALL_GCM_MODELS[event]) * len(ALL_IMPACT_MODELS[event]) + 1,
                    d_bin,
                )
            ) if event != "burntarea" else np.sort(np.arange(0, 13, 1,))
            # define discrete colormap
            norm = matplotlib.colors.BoundaryNorm(bins, cmap.N)
            self.log.info(
                f"Creating plot for {event}-{event1} for total "
                "and global time correlation..."
            )
            font_size = 9.5
            plt.rcParams.update({"font.size": font_size})
            fig, axis = plt.subplots(
                figsize=(6, 3), subplot_kw={"projection": ccrs.Robinson()}
            )
            axis.coastlines(color="grey")
            if self.use_result_median:
                model_counts_total = data.to_array(dim="t0").where(data.to_array(dim="t0") > 0).median("t0")
                model_counts_total = model_counts_total.where(model_counts_total > 0)
            else:
                model_counts_total = data.to_array(dim="t0").count(["t0", "model"])
                model_counts_total = model_counts_total.where(data.to_array(dim="t0").notnull().sum(["t0", "model"]))
            xr.plot.pcolormesh(
                darray=model_counts_total,
                cmap=cmap,
                norm=norm,
                transform=ccrs.PlateCarree(),
                ax=axis,
                add_colorbar=False,
                rasterized=True,
            )
            axis.set_global()
            # add cbar
            axis2 = fig.add_axes([0.9, 0.15, 0.03, 0.7])
            cbar2 = matplotlib.colorbar.ColorbarBase(
                axis2,
                cmap=cmap,
                norm=norm,
                spacing="uniform",
                ticks=bins,
                boundaries=bins,
                format="%.1f",
            )
            cbar2.set_ticks(ticks=bins, labels=bins)
            cbar2.ax.set_ylabel(
                "Number of regularity detecting models", rotation=90, labelpad=20
            )
            # remove frame and ticks
            axis.spines["top"].set_visible(False)
            axis.spines["right"].set_visible(False)
            axis.spines["bottom"].set_visible(False)
            axis.spines["left"].set_visible(False)
            axis.get_xaxis().set_ticks([])
            axis.get_yaxis().set_ticks([])
            if event == "cropfailedarea":
                figure_label = "a"
            elif event == "heatwavedarea":
                figure_label = "b"
            else:
                figure_label = "c"
            plt.text(
                0.09,
                0.9,
                figure_label,
                transform=plt.gcf().transFigure,
                horizontalalignment="left",
                verticalalignment="top",
                fontsize=2*font_size,
                fontweight='bold'
            )
            fig_title = f"{ISIMIP_IMPACT_NAME[self.impact_type]}, total"
            axis.set_title(fig_title)
            full_path = os.path.join(
                PLOTS_PATH, f"Nt_{NT}", f"{self.ssp_name}", "global_dynamic_plots"
            )
            if not os.path.exists(full_path):
                os.makedirs(full_path)
            no_trend_particle = "" if not no_trend else "_detrended"
            event_particle = (event, event1) if not RUN_CROP_TYPE_RESOLVED else (self.crop_type, self.crop_type)
            fig.savefig(
                os.path.join(
                    full_path,
                    f"total_{self.ssp_name}_"
                    f"{event_particle[0]}_median_median_{event_particle[1]}_median_median_modelcounts"
                    + no_trend_particle
                    + ".pdf",
                ),
                bbox_inches="tight",
                format='pdf',
            )
            # clear axis to add value plot
            plt.cla()
            plt.clf()
            plt.close()
            self.log.info("Plot finished!\n")

            for var in data.data_vars:
                tmp = f"[{int(var)}, {int(var) + NT * 2-1})"
                # determine limits of time window
                self.log.info(
                    f"Creating plot for {event}-{event1} for {tmp} "
                    "and global time correlation..."
                )
                font_size = 9.5
                plt.rcParams.update({"font.size": font_size})
                fig, axis = plt.subplots(
                    figsize=(6, 3), subplot_kw={"projection": ccrs.Robinson()}
                )
                axis.coastlines(color="grey")
                model_counts = (
                    data.where(data > 0)[var] if self.use_result_median
                    else data.where(data > 0).sum("model", skipna=False)[var]
                )
                xr.plot.pcolormesh(
                    darray=model_counts,
                    cmap=cmap,
                    norm=norm,
                    transform=ccrs.PlateCarree(),
                    ax=axis,
                    add_colorbar=False,
                    rasterized=True
                )
                axis.set_global()
                # add cbar
                axis2 = fig.add_axes([0.9, 0.15, 0.03, 0.7])
                cbar2 = matplotlib.colorbar.ColorbarBase(
                    axis2,
                    cmap=cmap,
                    norm=norm,
                    spacing="uniform",
                    ticks=bins,
                    boundaries=bins,
                    format="%.1f",
                )
                cbar2.set_ticks(ticks=bins, labels=bins)
                cbar2.ax.set_ylabel(
                    "Number of regularity detecting models", rotation=90, labelpad=20
                )
                # remove frame and ticks
                axis.spines["top"].set_visible(False)
                axis.spines["right"].set_visible(False)
                axis.spines["bottom"].set_visible(False)
                axis.spines["left"].set_visible(False)
                axis.get_xaxis().set_ticks([])
                axis.get_yaxis().set_ticks([])
                if int(var) == 1950:
                    figure_label = "a"
                elif int(var) == 2050 and not no_trend:
                    figure_label = "b"
                else:
                    figure_label = "c"
                plt.text(
                    0.09,
                    0.9,
                    figure_label,
                    transform=plt.gcf().transFigure,
                    horizontalalignment="left",
                    verticalalignment="top",
                    fontsize=2*font_size,
                    fontweight='bold'
                )
                fig_title = f"{ISIMIP_IMPACT_NAME[self.impact_type]}, {int(var)} - {int(var) + NT * 2 - 1}"
                if int(var) == 2050 and no_trend:
                    fig_title = fig_title + " (detrended)"
                axis.set_title(fig_title)
                full_path = os.path.join(
                    PLOTS_PATH, f"Nt_{NT}", f"{self.ssp_name}", "global_dynamic_plots"
                )
                if not os.path.exists(full_path):
                    os.makedirs(full_path)
                no_trend_particle = "" if not no_trend else "_detrended"
                event_particle = (event, event1) if not RUN_CROP_TYPE_RESOLVED else (self.crop_type, self.crop_type)
                fig.savefig(
                    os.path.join(
                        full_path,
                        f"{var}_{self.ssp_name}_"
                        f"{event_particle[0]}_median_median_{event_particle[1]}_median_median_modelcounts"
                        + no_trend_particle
                        + ".pdf",
                    ),
                    bbox_inches="tight",
                    format='pdf',
                )
                # clear axis to add value plot
                plt.cla()
                plt.clf()
                plt.close()
                self.log.info("Plot finished!\n")


def set_up_parser() -> Union[None, argparse.Namespace]:
    """Set up parser for argument parsing and return flags"""
    parser = argparse.ArgumentParser()
    # define flags
    parser.add_argument(
        "-nl",
        "--no_logfile",
        action="store_true",
        help="No logging in log file",
        default=True,
    )
    parser.add_argument(
        "-wildfire_cap",
        "--wildfire_cap",
        help="Specify the applied wildfire cap",
        nargs="?",
        type=float,
        default=-999,
    )
    parser.add_argument(
        "-r2_test",
        "--r2_test",
        action="store_true",
        help="This plot is part of the R2 test",
        default=False,
    )
    parser.add_argument(
        "-ssp",
        "--ssp_scenario",
        help="name of the ssp scenario",
        type=str,
        required=True,
    )
    parser.add_argument(
        "-impact",
        "--impact_type",
        help="name of the extreme event type",
        type=str,
        required=True,
    )
    parser.add_argument(
        "-crop",
        "--crop_type",
        help="name of crop type in case of type specific calculation",
        type=str,
        required=RUN_CROP_TYPE_RESOLVED,
    )
    return parser.parse_args()


# pylint: disable=too-many-statements
def main():
    """Main function that plots time series analysis output"""
    flags = set_up_parser()
    # log start of simulation with current time
    start = datetime.now(pytz.timezone("UTC"))
    # set up log output form (stdout or file), by default: only stderr
    log_handlers = [logging.StreamHandler(sys.stdout)]
    if not flags.no_logfile:
        # create a name for the log file based on the start time
        log_filename = (
            start.strftime("%Y_%m_%d_%H_%M_%S") + "return_period_calculation.log"
        )
        # include log file
        log_handlers.append(
            logging.FileHandler(os.path.join(os.getcwd(), LOG_PATH, log_filename))
        )
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(message)s",
        level=logging.INFO,
        handlers=log_handlers,
    )
    if USE_RESULT_MEDIAN and USE_MODEL_MEAN:
        logging.error("You cannot use the result AND model mean at the same time!")
        raise ValueError

    logging.info("Starting execution of time series plots")
    logging.info("==========================================")
    # pylint:disable=logging-fstring-interpolation
    logging.info(f"SSP={ALL_SSP_SCENARIOS[flags.ssp_scenario]}")
    logging.info(f"Impact type={flags.impact_type}")
    logging.info(f"NT={NT}")
    logging.info(f"wildfire cap={flags.wildfire_cap}")
    logging.info(f"Reference times={t_0s}")
    logging.info("==========================================")
    # initialize time series analysis object
    plot_time_analysis = PlotTimeAnalysisImpacts(
        impact_type=flags.impact_type,
        use_all_gcms=USE_ALL_GCM_MODELS,
        use_mean=USE_MODEL_MEAN,
        use_result_median=USE_RESULT_MEDIAN,
        ssp=ALL_SSP_SCENARIOS[flags.ssp_scenario],
        log=logging,
        is_r2test=flags.r2_test,
        wildfire_cap=flags.wildfire_cap,
        crop_type=flags.crop_type,
    )
    if (
        not plot_time_analysis.use_result_median
        and not plot_time_analysis.use_model_mean
        or plot_time_analysis.use_model_mean
    ):
        if plot_time_analysis.impact_time_series is None:
            return

        # create local dynamic plots
        logging.info("Creating local plots...")
        start = time.time()
        plot_time_analysis.create_local_impact_plots()
        plot_time_analysis.create_local_correlation_plots(True)
        plot_time_analysis.create_local_correlation_plots(False)
        logging.info(
            f"Local plots successfully created in {(time.time() - start) / 60} minutes!"
        )

        # create global static plots
        logging.info("Creating global static plots...")
        start = time.time()
        plot_time_analysis.create_global_plots()
        logging.info(
            f"Global static plots successfully created in "
            f"{(time.time() - start) / 60} minutes!"
        )

    if not RUN_CROP_TYPE_RESOLVED:
        # create counting time series plots
        logging.info("Creating counting time series plots...")
        start = time.time()
        plot_time_analysis.create_counting_time_plots()
        logging.info(
            "Counting time series plots successfully created in "
            f"{(time.time() - start) / 60} minutes!"
        )

    # create model count for dominant return period plots
    logging.info("Creating model count for dominant return period plots...")
    start = time.time()

    plot_time_analysis.create_model_count_plots(no_trend=False)
    plot_time_analysis.create_model_count_plots(no_trend=True)
    logging.info(
        f"Model count for dominant return period plots successfully created in "
        f"{(time.time() - start) / 60} minutes!"
    )

    # create impact count plots
    logging.info("Creating impact count plots...")
    start = time.time()
    plot_time_analysis.create_impact_count_plots()
    logging.info(
        f"Impact count plots successfully created in "
        f"{(time.time() - start) / 60} minutes!"
    )

    # create separate global correlation plots
    logging.info("Creating separate global correlation plots...")
    start = time.time()
    plot_time_analysis.create_separate_correlation_plots(True, "t_0")
    plot_time_analysis.create_separate_correlation_plots(False, "t_0")
    if plot_time_analysis.use_result_median:
        plot_time_analysis.create_separate_correlation_plots(True, "model")
        plot_time_analysis.create_separate_correlation_plots(False, "model")
        plot_time_analysis.create_separate_correlation_plots(True, "total")
        plot_time_analysis.create_separate_correlation_plots(False, "total")
        plot_time_analysis.create_separate_correlation_plots_std(False, "total")
    logging.info(
        f"Global separate correlation plots successfully created in "
        f"{(time.time() - start) / 60} minutes!"
    )


# Press the green button in the gutter to run the script.
if __name__ == "__main__":
    main()
