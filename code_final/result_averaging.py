"""
This script calculates the average of the time series results (main.py).
For more details check gitlab project.

Author: Karim Zantout
"""

import argparse
import glob
import logging
from datetime import datetime
import os
import csv
import sys
from typing import Union
import warnings
import numpy as np
import xarray as xr
import pytz
import pandas as pd
from settings import OUTPUT_PATH, TEST_OUTPUT_PATH, LOG_PATH, ALL_IMPACT_MODELS


# pylint: disable=too-many-locals
def read_data(
    data_type: str, no_trend: bool, is_test: bool, log: logging
) -> (dict, dict):
    """Read impacts data of specific event type, perform statistics and store"""
    path = TEST_OUTPUT_PATH if is_test else OUTPUT_PATH
    # sub path for return periods
    if data_type == "dominant_return_period" and no_trend:
        sub_path = "detrended"
    elif data_type == "dominant_return_period" and not no_trend:
        sub_path = "original"
    else:
        sub_path = ""
    # get all impact combinations
    subdirectories = [
        directory
        for directory in os.listdir(os.path.join(path, sub_path, data_type))
        if os.path.isdir(
            os.path.join(os.path.join(path, sub_path, data_type), directory)
        )
    ]
    res = {}
    res_model = {}
    for impact in subdirectories:
        filenames = glob.glob(
            os.path.join(path, sub_path, data_type, impact, "*", "*.nc"),
        )
        # sort result files to ssp and n_t
        res[impact] = {}
        res_model[impact] = {}
        for filename in filenames:
            if not filename.endswith(".nc"):
                log.error(
                    "File name extension is expected to be .nc "
                    f"but got {filename.split('.')[-1]} instead"
                )
                raise ValueError
            # remove filetype ending and split
            _filename = os.path.basename(filename).split(".")[0].split("_")
            # make sure that only valid impact models are included
            if data_type == "dominant_return_period":
                if (
                    filename.split(os.sep)[-2].split("_")[0]
                    not in ALL_IMPACT_MODELS[impact.split("_")[0]]
                    or filename.split(os.sep)[-2].split("_")[1]
                    not in ALL_IMPACT_MODELS[impact.split("_")[1]]
                ):
                    continue
            else:
                if (
                    filename.split(os.sep)[-2].split("_")[0]
                    not in ALL_IMPACT_MODELS[impact.split("_")[0]]
                ):
                    continue
            # get ssp and n_t from filename
            n_t = (
                _filename[-3]
                if data_type == "dominant_return_period"
                else _filename[-4]
            )
            ssp = (
                _filename[-6]
                if data_type == "dominant_return_period"
                else _filename[-7]
            )
            model_name = _filename[3]
            if (ssp, n_t) not in res[impact]:
                res[impact][(ssp, n_t)] = []
            if (ssp, n_t, model_name) not in res_model[impact]:
                res_model[impact][(ssp, n_t, model_name)] = []
            log.info(f"Reading file {filename}...")
            # append data into a data set for each extreme event
            data = xr.open_dataset(filename)
            res[impact][(ssp, n_t)].append(data)
            res_model[impact][(ssp, n_t, model_name)].append(data)
            log.info(f"{filename} successfully parsed!\n")
    return res, res_model


# pylint: disable=too-many-locals
def do_statistics(
    data_type: str, no_trend: bool, is_test: bool, log: logging
) -> (dict, dict, dict, dict, dict, dict, dict, dict, dict, dict, dict):
    """Calculates statistics for specific data_type"""
    data_dict, data_model_dict = read_data(data_type, no_trend, is_test, log)
    # nan-median along all impact and climate models (for all t0s and positions)
    median = {}
    # nan-std along all impact and climate models (for all t0s and positions)
    std = {}
    # dominant return period: non-nan counts along all impact and climate models
    # (for all t0s and positions)
    # event counts: non-zero counts along all impact and climate models (for all t0s and positions)
    non_nan_counts = {}
    # dominant return period: non-nan counts along all t0s, positions, impact and climate models
    # event counts: non-zero counts along all t0s, positions, impact and climate models
    total_non_nan_counts = {}
    # nan-median along all t0s and impact models (for all climate models and positions)
    median_model = {}
    # nan-std along all t0s and impact models (for all climate models and positions)
    std_model = {}
    # nan-median along all t0s, climate and impact models (for all positions)
    median_total = {}
    # nan-std along all t0s, climate and impact models (for all positions)
    std_total = {}
    # nan-median along all t0s, positions, climate and impact models
    total_median = {}
    # nan-mean along all t0s, positions, climate and impact models
    total_mean = {}
    # nan-std along all t0s, positions, climate and impact models
    total_std = {}
    for impact, data_val in data_dict.items():
        # concat data into a single large dataframe
        log.info(f"Calculate statistics from data for {impact}...")
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            # statistics over climate and impact model
            median[impact] = {
                key: xr.concat(data, dim="model_type").median(dim="model_type")
                for key, data in data_val.items()
            }
            std[impact] = {
                key: xr.concat(data, dim="model_type").std(dim="model_type")
                for key, data in data_val.items()
            }
            non_nan_counts[impact] = {
                key: xr.concat(data, dim="model_type")
                .where(xr.concat(data, dim="model_type") > 0)
                .count(dim="model_type")
                for key, data in data_val.items()
            }
            total_non_nan_counts[impact] = {
                key: np.sum(xr.concat(data, dim="model_type").to_array() > 0)
                for key, data in data_val.items()
            }
            # statistics over t_0 and impact model
            median_model[impact] = {
                key: xr.concat(data, dim="model_type")
                .to_array(dim="t_0")
                .median(dim=["t_0", "model_type"])
                for key, data in data_model_dict[impact].items()
            }
            # combine models into a single data set
            median_model[impact] = {
                key: xr.Dataset(
                    {
                        var[2]: median_model[impact][(key[0], key[1], var[2])]
                        for var in median_model[impact].keys()
                        if key == var[:-1]
                    }
                )
                for key in median[impact].keys()
            }
            std_model[impact] = {
                key: xr.concat(data, dim="model_type")
                .to_array(dim="t_0")
                .std(dim=["t_0", "model_type"])
                for key, data in data_model_dict[impact].items()
            }
            std_model[impact] = {
                key: xr.Dataset(
                    {
                        var[2]: std_model[impact][(key[0], key[1], var[2])]
                        for var in std_model[impact].keys()
                        if key == var[:-1]
                    }
                )
                for key in std[impact].keys()
            }
            # statistics over t_0, climate and impact model
            median_total[impact] = {
                key: xr.concat(data, dim="model_type")
                .to_array(dim="t_0")
                .median(dim=["t_0", "model_type"])
                for key, data in data_val.items()
            }
            # create mask (majority of input along the time axis is not nan)
            majority_mask = {
                key: data.to_array(dim="t_0") for key, data in median[impact].items()
            }
            majority_mask = {
                key: val.notnull().sum(dim="t_0") >= len(val["t_0"].values) / 2
                for key, val in majority_mask.items()
            }
            # apply majority mask
            median_total[impact] = {
                key: xr.where(majority_mask[key], data, np.nan)
                for key, data in median_total[impact].items()
            }
            std_total[impact] = {
                key: xr.concat(data, dim="model_type")
                .to_array(dim="t_0")
                .std(dim=["t_0", "model_type"])
                for key, data in data_val.items()
            }
            # statistics over t_0, climate, location and impact model
            total_median[impact] = {
                key: float(
                    xr.concat(data, dim="model_type").to_array(dim="t_0").median()
                )
                for key, data in data_val.items()
            }
            total_mean[impact] = {
                key: float(xr.concat(data, dim="model_type").to_array(dim="t_0").mean())
                for key, data in data_val.items()
            }
            total_std[impact] = {
                key: float(xr.concat(data, dim="model_type").to_array(dim="t_0").std())
                for key, data in data_val.items()
            }
        log.info(f"Statistics for {impact} successfully calculated!")
    return (
        median,
        std,
        non_nan_counts,
        total_non_nan_counts,
        median_model,
        std_model,
        median_total,
        std_total,
        total_median,
        total_mean,
        total_std,
    )


# pylint: disable=too-many-branches
def calc_and_store_statistics(
    data_type: str, no_trend: bool, is_test: bool, log: logging
) -> None:
    """Stores statistics to files"""
    (
        median,
        std,
        non_nan_counts,
        total_non_nan_counts,
        median_model,
        std_model,
        median_total,
        std_total,
        total_median,
        total_mean,
        total_std,
    ) = do_statistics(data_type, no_trend, is_test, log)
    path = TEST_OUTPUT_PATH if is_test else OUTPUT_PATH
    # sub path for return periods
    if data_type == "dominant_return_period" and no_trend:
        sub_path = "detrended"
    elif data_type == "dominant_return_period" and not no_trend:
        sub_path = "original"
    else:
        sub_path = ""
    for impact, median_val in median.items():
        log.info("Storing data...")
        if data_type == "dominant_return_period" and not no_trend:
            no_trend_particle = "original"
            path_name = os.path.join(path, "statistical_test")
            with open(
                os.path.join(
                    path_name,
                    f"total_median_{no_trend_particle}_{data_type}_{impact}.csv",
                ),
                "w",
                encoding="utf-8",
            ) as csv_file:
                writer = csv.writer(csv_file)
                for key, value in total_median[impact].items():
                    writer.writerow([key[0], key[1], value])
            with open(
                os.path.join(
                    path_name,
                    f"total_mean_{no_trend_particle}_{data_type}_{impact}.csv",
                ),
                "w",
                encoding="utf-8",
            ) as csv_file:
                writer = csv.writer(csv_file)
                for key, value in total_mean[impact].items():
                    writer.writerow([key[0], key[1], value])
            with open(
                os.path.join(
                    path_name,
                    f"total_std_{no_trend_particle}_{data_type}_{impact}.csv",
                ),
                "w",
                encoding="utf-8",
            ) as csv_file:
                writer = csv.writer(csv_file)
                for key, value in total_std[impact].items():
                    writer.writerow([key[0], key[1], value])
            with open(
                os.path.join(
                    path_name,
                    f"total_signals_{no_trend_particle}_{data_type}_{impact}.csv",
                ),
                "w",
                encoding="utf-8",
            ) as csv_file:
                writer = csv.writer(csv_file)
                for key, value in total_non_nan_counts[impact].items():
                    writer.writerow(
                        [
                            key[0],
                            key[1],
                            len(non_nan_counts[impact][key].data_vars),
                            int(value),
                        ]
                    )
        path_name = os.path.join(path, sub_path, data_type, impact)
        for key, data in non_nan_counts[impact].items():
            data.to_netcdf(
                os.path.join(path_name, f"{impact}_modelcounts_{key[0]}_{key[1]}_t0.nc")
            )
        for key, data in median_val.items():
            data.to_netcdf(
                os.path.join(
                    path_name,
                    f"{impact}_median_{key[0]}_{key[1]}_t0.nc",
                )
            )
        for key, data in std[impact].items():
            data.to_netcdf(
                os.path.join(
                    path_name,
                    f"{impact}_std_{key[0]}_{key[1]}_t0.nc",
                )
            )
        for key, data in median_model[impact].items():
            data.to_netcdf(
                os.path.join(
                    path_name,
                    f"{impact}_median_{key[0]}_{key[1]}_model.nc",
                )
            )
        for key, data in std_model[impact].items():
            data.to_netcdf(
                os.path.join(
                    path_name,
                    f"{impact}_std_{key[0]}_{key[1]}_model.nc",
                )
            )
        for key, data in median_total[impact].items():
            data.to_netcdf(
                os.path.join(
                    path_name,
                    f"{impact}_median_{key[0]}_{key[1]}_total.nc",
                )
            )
        for key, data in std_total[impact].items():
            data.to_netcdf(
                os.path.join(
                    path_name,
                    f"{impact}_std_{key[0]}_{key[1]}_total.nc",
                )
            )
        log.info("Data successfully stored!")


def read_csv_statistics(is_test: bool, log: logging) -> dict:
    """Read impacts data of csv type, perform statistics and store"""
    path = TEST_OUTPUT_PATH if is_test else OUTPUT_PATH
    # get all impact combinations
    subdirectories = [
        directory
        for directory in os.listdir(os.path.join(path, "event_counts"))
        if os.path.isdir(os.path.join(os.path.join(path, "event_counts"), directory))
    ]
    full_data = {}
    for impact in subdirectories:
        filenames = glob.glob(
            os.path.join(path, "event_counts", impact, "*", "*.csv"),
        )
        # sort result files to ssp and n_t
        full_data[impact] = pd.DataFrame()
        for filename in filenames:
            if not filename.endswith(".csv"):
                log.error(
                    "File name extension is expected to be .csv "
                    f"but got {filename.split('.')[-1]} instead"
                )
                raise ValueError
            # remove filetype ending and split
            _filename = os.path.basename(filename).split(".")[0].split("_")
            # make sure that impact model is correct
            if _filename[0] not in ALL_IMPACT_MODELS[impact]:
                continue
            # get ssp and n_t from filename
            n_t = _filename[-5]
            ssp = _filename[-8]
            gcm = _filename[-6]
            log.info(f"Reading file {filename}...")
            # append data into a data set for each extreme event
            data = pd.read_csv(filename).set_index("year")
            data.index = pd.MultiIndex.from_tuples(
                [(year, ssp, n_t, gcm) for year in data.index],
                names=["year", "ssp", "Nt", "GCM"],
            )
            full_data[impact] = pd.concat([full_data[impact], data])
            log.info(f"{filename} successfully parsed!\n")
    return full_data


def do_csv_statistics(
    is_test: bool, log: logging
) -> (dict, dict, dict, dict, dict, dict, dict):
    """Calculates statistics from csv data"""
    full_data = read_csv_statistics(is_test, log)
    median = {}
    std = {}
    median_gcm = {}
    mean_gcm = {}
    min_gcm = {}
    max_gcm = {}
    std_gcm = {}
    for impact, data in full_data.items():
        log.info(f"Calculate statistics for {impact} data...")
        median[impact] = data.groupby(["year", "ssp", "Nt"]).median()
        median_gcm[impact] = data.groupby(["year", "ssp", "Nt", "GCM"]).median()
        mean_gcm[impact] = data.groupby(["year", "ssp", "Nt", "GCM"]).mean()
        min_gcm[impact] = data.groupby(["year", "ssp", "Nt", "GCM"]).min()
        max_gcm[impact] = data.groupby(["year", "ssp", "Nt", "GCM"]).max()
        std[impact] = data.groupby(["year", "ssp", "Nt"]).std()
        std_gcm[impact] = data.groupby(["year", "ssp", "Nt", "GCM"]).std()
        log.info(f"Statistics for {impact} successfully calculated!")
    return median, std, median_gcm, mean_gcm, min_gcm, max_gcm, std_gcm


def calc_and_store_csv_statistics(is_test: bool, log: logging) -> None:
    """Calculates and stores csv data statistics"""
    path = TEST_OUTPUT_PATH if is_test else OUTPUT_PATH
    median, std, median_gcm, mean_gcm, min_gcm, max_gcm, std_gcm = do_csv_statistics(
        is_test, log
    )
    for impact, median_val in median.items():
        log.info(f"Storing {impact} data...")
        for (ssp, n_t), data in median_val.groupby(["ssp", "Nt"]):
            for gcm in median_gcm[impact].index.get_level_values("GCM").unique():
                median_gcm[impact].loc[(slice(None), ssp, n_t, gcm), :].to_csv(
                    os.path.join(
                        path,
                        "event_counts",
                        impact,
                        f"{impact}_{gcm}_median_{ssp}_{n_t}.csv",
                    )
                )
                mean_gcm[impact].loc[(slice(None), ssp, n_t, gcm), :].to_csv(
                    os.path.join(
                        path,
                        "event_counts",
                        impact,
                        f"{impact}_{gcm}_mean_{ssp}_{n_t}.csv",
                    )
                )
                min_gcm[impact].loc[(slice(None), ssp, n_t, gcm), :].to_csv(
                    os.path.join(
                        path,
                        "event_counts",
                        impact,
                        f"{impact}_{gcm}_min_{ssp}_{n_t}.csv",
                    )
                )
                max_gcm[impact].loc[(slice(None), ssp, n_t, gcm), :].to_csv(
                    os.path.join(
                        path,
                        "event_counts",
                        impact,
                        f"{impact}_{gcm}_max_{ssp}_{n_t}.csv",
                    )
                )
            data.to_csv(
                os.path.join(
                    path,
                    "event_counts",
                    impact,
                    f"{impact}_median_{ssp}_{n_t}.csv",
                )
            )
        for (ssp, n_t), data in std[impact].groupby(["ssp", "Nt"]):
            for gcm in std_gcm[impact].index.get_level_values("GCM").unique():
                std_gcm[impact].loc[(slice(None), ssp, n_t, gcm), :].to_csv(
                    os.path.join(
                        path,
                        "event_counts",
                        impact,
                        f"{impact}_{gcm}_std_{ssp}_{n_t}.csv",
                    )
                )
            data.to_csv(
                os.path.join(
                    path, "event_counts", impact, f"{impact}_std_{ssp}_{n_t}.csv"
                )
            )
        log.info(f"Data for {impact} successfully stored!")


def create_model_statistics(log: logging) -> None:
    """calculates statistics from all time analysis results"""
    calc_and_store_csv_statistics(False, log)
    calc_and_store_statistics("dominant_return_period", True, False, log)
    calc_and_store_statistics("dominant_return_period", False, False, log)
    calc_and_store_statistics("event_counts", False, False, log)


def set_up_parser() -> Union[None, argparse.Namespace]:
    """Set up parser for argument parsing and return flags"""
    parser = argparse.ArgumentParser()
    # define flags
    parser.add_argument(
        "-nl",
        "--no_logfile",
        action="store_true",
        help="No logging in log file",
    )
    return parser.parse_args()


def main() -> None:
    """main function of script which creates model medians for each event type separately"""
    flags = set_up_parser()
    # log start of simulation with current time
    start = datetime.now(pytz.timezone("UTC"))
    # set up log output form (stdout or file), by default: only stderr
    log_handlers = [logging.StreamHandler(sys.stdout)]
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(message)s",
        level=logging.INFO,
        handlers=log_handlers,
    )
    logging.info("Starting execution of result averaging")
    logging.info("Calculating model statistics...")
    create_model_statistics(logging)
    logging.info("Calculation of model statistics successful!")
    end = datetime.now(pytz.timezone("UTC"))
    logging.info("Thank you for running the code! \n Total time: %s", str(end - start))


# Press the green button in the gutter to run the script.
if __name__ == "__main__":
    main()
