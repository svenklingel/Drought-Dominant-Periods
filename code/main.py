"""
This script calculates time convolutions of yearly ISIMIP extreme event time series data
and counts the number of extreme events in time bins (DATA_INPUT_PATH)
For more details check gitlab project.

Author: Karim Zantout
"""

import argparse
from datetime import datetime
from typing import Union
import glob
import csv
import os
import scipy.stats
import logging
import warnings
import sys 
import pytz
import numpy as np
import xarray as xr
from sklearn.metrics import r2_score
from util import surface_area
from settings import (
    SINGLE_GCM_MODEL,
    SINGLE_IMPACT_MODEL,
    USE_ALL_GCM_MODELS,
    USE_ALL_IMP_MODELS,
    ALL_SSP_SCENARIOS,
    ALL_GCM_MODELS,
    ALL_IMPACT_MODELS,
    OUTPUT_PATH,
    LOG_PATH,
    RUN_DOMINANT_FREQUENCY_CALC,
    NT,
    LOCATIONS,
    t_0s,
    EPS_CORR,
    USE_MODEL_MEAN,
    INPUT_DATA_PATH,
    DATA_MEAN_PATH,
    R2_THRESHOLD,
)


def _calc_time_corr(data: xr.Dataset, event: str, event1: str) -> np.array:
    """
    Calculates time correlation between event and event1. The data is stored
    in x.
    :param data: Dataset with time-sorted data
    :param event: name of extreme event
    :param event1: name of other extreme event
    :return: time correlation array
    """
    # no need to calculate overlap if the left array is zero
    f_i = data[event].isel(time=range(NT)).values
    if list(set(f_i)) == [0.0]:
        return np.array([0.0] * NT)
    # compute time correlation function
    f_j = data[event1].isel(time=range(2 * NT)).values
    if list(set(f_j)) == [0.0]:
        return np.array([0.0] * NT)
    f_j = np.array([f_j[n_shift : n_shift + NT] for n_shift in range(NT)]).reshape(
        NT, NT
    )
    return f_i.dot(f_j.T) / NT


def _calc_time_corr_array(data: xr.DataArray, data1: xr.DataArray) -> xr.DataArray:
    """
    Calculates time correlation between event and event1. The data is stored
    in x.
    :param data: Dataset with time-sorted data
    :return: time correlation array
    """
    # no need to calculate overlap if the left array is zero
    f_i = data.isel(time=range(NT)).values
    # compute time correlation function
    f_j = data1.isel(time=range(2 * NT)).values
    f_j = np.array([f_j[n_shift : n_shift + NT] for n_shift in range(NT)])
    return xr.DataArray(
        data=np.einsum("ijk,lijk->ljk", f_i, f_j) / NT,
        dims=data.dims,
        coords=data.isel(time=range(NT)).coords,
        attrs={"standard_name": "time_correlation", "unit": "1"},
    ).assign_coords(time=range(NT))


def calc_chi2_significance(
    data: np.array, fourier_power: np.array, idx_dominant_period: int
) -> bool:
    """
    Calculates test result of chi2 test for spectral significance of
    dominant period
    """
    # make sure that this candidate is also significant through chi2 test
    # calculate significance compared to red noise
    # construct expected red noise spectrum at candidate frequency
    # see Torrence and Campo 1998 Eq. (17) and Zhang and Moore 2011
    # Percival and Walden 1993
    lin_reg = np.polyfit(np.arange(NT), data, 1)
    detrended_data = data - (lin_reg[0] * np.arange(NT) + lin_reg[1])
    phi = detrended_data - np.mean(detrended_data)
    phi = phi[:NT - 1].dot(phi[1: NT]) / ((NT - 1) * np.var(data[:-1]))
    # red noise coefficient has to be positive (else blue noise)
    if phi < 0:
        return True
    rspec = np.zeros(len(fourier_power))
    for idx in range(len(fourier_power)):
        rspec[idx] = (1. - phi ** 2) / (
                1. - 2. * phi * np.cos(2 * np.pi * idx / NT) + phi ** 2
        )
    # normalize spectra for comparison with each other
    # normalize with de-meaned variance which equals sum of power
    # according to Parseval's theorem
    rspec = rspec / np.sum(rspec)
    # de-meaning = set zero coefficient to zero
    power_spectrum = fourier_power.copy()
    power_spectrum = np.nan_to_num(power_spectrum / np.sum(power_spectrum), nan=0)
    # get spectral power density at dominant period
    rspec_dominant_period = rspec[idx_dominant_period]
    power_spectrum_dominant_period = power_spectrum[idx_dominant_period]
    # we use the 95% significance level for chi2 test with 2 dof
    chi2_stat = scipy.stats.chi2.ppf(.95, 2)
    # scale red noise according to significance level
    spec95 = np.nan_to_num(chi2_stat * rspec_dominant_period, nan=0)
    return power_spectrum_dominant_period >= spec95


def calc_chi2_significance_array(
    data: np.array, fourier_power: np.array, idx_dominant_period: np.array
) -> np.array:
    """
    Calculates test result of chi2 test for spectral significance of
    dominant period
    """
    # calculate significance compared to red noise
    # construct expected red noise spectrum at candidate frequency
    # see Torrence and Campo 1998 Eq. (17) and Zhang and Moore 2011
    # Percival and Walden 1993
    if "degree" in data.coords.keys():
        data = data.drop_vars("degree")
    lin_reg = data.polyfit("time", 1)
    reversed_coords = list(data.coords.keys())
    reversed_coords.reverse()
    detrended_data = data - (
        lin_reg.sel(degree=0)["polyfit_coefficients"]
        + lin_reg.sel(degree=1)["polyfit_coefficients"] * data.time
    ).transpose(*reversed_coords)
    phi = detrended_data - np.mean(detrended_data, axis=0)
    phi = np.einsum(
        "ijk,ijk->jk",
        phi.isel(time=range(NT - 1)),
        phi.isel(time=range(1, NT))
    ) / ((NT - 1) * data.isel(time=range(NT-1)).var(dim="time"))
    rspec = np.zeros_like(np.abs(fourier_power))
    for h in range(fourier_power.shape[0]):
        rspec[h] = (1. - phi ** 2) / (1. - 2. * phi * np.cos(2 * np.pi * h / NT) + phi ** 2)
    # normalize with de-meaned variance which equals sum of power
    # according to Parseval's theorem
    rspec = rspec / (rspec.sum(axis=0)[None, :, :])
    power_spectrum = fourier_power.copy()
    power_spectrum = np.nan_to_num(
        power_spectrum / (power_spectrum.sum(axis=0)[None, :, :]),
        nan=0
    )
    rspec_dominant_period = np.zeros_like(rspec[0])
    power_spectrum_dominant_period = np.zeros_like(power_spectrum[0])
    for i in range(idx_dominant_period.shape[0]):
        for j in range(idx_dominant_period.shape[1]):
            for val in idx_dominant_period[i, j]["idx"]:
                rspec_dominant_period[i, j] = rspec[val, i, j]
                power_spectrum_dominant_period[i, j] = power_spectrum[val, i, j]
    # we use the 95% significance level for chi2 test with 2 dof
    chi2_stat = scipy.stats.chi2.ppf(.95, 2)
    # scale red noise according to dof
    spec95 = np.nan_to_num(chi2_stat * rspec_dominant_period, nan=0)
    # red noise coefficient has to be positive (else blue noise)
    return np.logical_or(power_spectrum_dominant_period >= spec95, phi < 0)


# pylint: disable=too-many-branches
def _determine_dominant_return_period(
    corr: list,
) -> (float, np.array, Union[float, None], np.array, np.array):
    """
    Determines the dominant return period of a Fourier series and returns dominant
    return period if there is a significant dominant return period and 0.0 else. For the
    definition of a dominant return period, see note.
    :param corr: correlation data

    :return res: R² of fit, fit, significant dominant return period,
    Fourier coefficients, and list of the largest Fourier indices
    """
    # compute Fourier series coefficients and power spectrum
    c_n = np.fft.rfft(corr)
    V_n = (c_n*c_n.conj()).real 
    # non-zero frequency are doubled (complex conjugated frequencies)
    V_n[1:] = V_n[1:]*2
    # sort indices of coefficients in increasing order
    sorted_idx = np.flip(np.argpartition(np.abs(c_n), range(-len(c_n), 0)))
    # check if corr is significant
    if np.max(np.abs(corr)) < EPS_CORR:
        return 1.0, corr, None, c_n, sorted_idx
    # check if corr is not constant
    if np.sum(np.abs(c_n[1:])) < 1e-10:
        return 1.0, corr, 1, c_n, sorted_idx
    # determine the smallest candidate return period which is the gcd of
    # the two largest non-zero coefficients
    largest_nonzero_coeff_indices = sorted_idx[sorted_idx != 0]
    gcd = np.gcd(largest_nonzero_coeff_indices[0], largest_nonzero_coeff_indices[1])
    # if gcd == 1 we set it to the return period corresponding to
    # the maximum non-zero coefficient
    if gcd == 1 and largest_nonzero_coeff_indices[0] > 1:
        gcd = largest_nonzero_coeff_indices[0]
    elif gcd == 1:
        gcd = largest_nonzero_coeff_indices[1]
    # collect largest coefficients that are multiples of each other
    tmp = []
    for idx in sorted_idx:
        # zero return period is fine
        if idx == 0:
            tmp.append(idx)
        # check whether return period is multiple
        elif (idx % gcd == 0) or (gcd % idx == 0):
            tmp.append(idx)
        # if the current idx is not a multiple stop the search
        else:
            break
    # remove elements until order of harmonics is increasing
    tmp = np.array(tmp)
    while any(tmp[tmp != 0] != sorted(tmp[tmp != 0])):
        tmp = tmp[:-1]
    # make sure that gcd is also in list of the largest indices
    if gcd not in tmp:
        return 0, [0.0] * NT, None, c_n, sorted_idx
    # return smallest return period that is non-zero and where
    # the fit fine
    tmp_non_zero = [val for val in tmp if val != 0]

    significant_dominant_period = calc_chi2_significance(corr, V_n, tmp_non_zero[0])
    if not significant_dominant_period:
        return 0, [0.0] * NT, None, c_n, sorted_idx

    fit_n = [0.0 + 1j * 0.0] * len(c_n)
    # replace zero by coefficient at proper position
    for val in tmp:
        fit_n[val] = c_n[val]
    # do inverse Fourier transformation to get fit
    fourier_fit = np.fft.irfft(fit_n, len(corr))
    r2_val = 1 - (1 - r2_score(corr, fourier_fit)) * (len(corr) - 1) / (
        len(corr) - len(tmp) - 1
    )
    # the return period can be calculated via f=Nt/idx because the inverse
    # Fourier transform is given by
    # corr[n] = c_0 + \sum_{idx} [
    # 2*Re(c_{idx})*cos(2*pi*idx*n/Nt) - 2*Im(c_{idx})*sin(2*pi*idx*n/Nt) ]
    if r2_val >= R2_THRESHOLD:
        return (
            r2_val,
            fourier_fit,
            np.round(NT / tmp_non_zero[0], decimals=2),
            c_n,
            sorted_idx,
        )
    return r2_val, fourier_fit, None, c_n, sorted_idx


# pylint: disable=too-many-locals
def _determine_dominant_return_period_array(corr: xr.DataArray) -> xr.DataArray:
    """
    Determines the dominant return period of a Fourier series and returns dominant
    return period if there is a significant dominant return period and 0.0 else. For the
    definition of a dominant return period, see note.
    :param corr: correlation data

    :return res: significant dominant return period Fourier coefficients
    """
    res = np.array([None] * corr[0, :, :].size).reshape(corr.shape[1:])
    # compute Fourier series coefficients
    c_n = np.fft.rfft(corr, axis=0)
    V_n = (c_n*c_n.conj()).real
    # non-zero frequency are doubled (complex conjugated frequencies)
    V_n[1:, :, :] = V_n[1:, :, :]*2
    # sort indices of coefficients in increasing order
    sorted_idx = np.flip(np.argsort(np.abs(c_n), axis=0), axis=0)
    # determine the smallest candidate return period which is the gcd of
    # the two largest non-zero coefficients
    largest_nonzero_coeff_indices = np.apply_along_axis(
        lambda x: x[x != 0][:2], axis=0, arr=sorted_idx
    )
    gcd = np.gcd(
        largest_nonzero_coeff_indices[0, :, :], largest_nonzero_coeff_indices[1, :, :]
    )
    # if gcd == 1 we set it to the return period corresponding to
    # the maximum non-zero coefficient
    gcd[
        np.logical_and(gcd == 1, largest_nonzero_coeff_indices[0, :, :] > 1)
    ] = largest_nonzero_coeff_indices[0, :, :][
        np.logical_and(gcd == 1, largest_nonzero_coeff_indices[0, :, :] > 1)
    ]
    gcd[
        np.logical_and(gcd == 1, largest_nonzero_coeff_indices[0, :, :] <= 1)
    ] = largest_nonzero_coeff_indices[1, :, :][
        np.logical_and(gcd == 1, largest_nonzero_coeff_indices[0, :, :] <= 1)
    ]
    # collect largest coefficients that are multiples of each other
    # zero return period is always fine and check whether return period is multiple
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        tmp = np.logical_or(
            sorted_idx == 0, np.logical_or(sorted_idx % gcd == 0, gcd % sorted_idx == 0)
        )
    # find position where above conditions do not apply anymore
    tmp = np.apply_along_axis(
        lambda x: np.argmax(np.invert(x)),
        axis=0,
        arr=tmp,
    )
    # get indices that comply with above conditions
    tmp = np.array(
        [
            {"idx": sorted_idx[: tmp[i, j], i, j]}
            for i in range(tmp.shape[0])
            for j in range(tmp.shape[1])
        ],
        dtype=object,
    ).reshape(tmp.shape)

    # remove elements until order of harmonics is increasing
    def _keep_increasing_subarray(array: np.array) -> np.array:
        # make exception on c_0 which should be allowed to appear anywhere
        while any(array[array != 0] != sorted(array[array != 0])):
            array = array[:-1]
        return array

    tmp = np.array(
        [{"idx": _keep_increasing_subarray(val["idx"])} for val in tmp.flatten()],
        dtype=object,
    ).reshape(tmp.shape)
    # return smallest return period that is non-zero and where
    # the fit fine
    tmp_non_zero = np.array(
        [
            {"idx": tmp[i, j]["idx"][tmp[i, j]["idx"] != 0]}
            for i in range(tmp.shape[0])
            for j in range(tmp.shape[1])
        ],
        dtype=object,
    ).reshape(tmp.shape)

    significance_mask = calc_chi2_significance_array(corr, V_n, tmp_non_zero)

    fit_n = (np.array([0.0 + 1j * 0.0] * c_n.size)).reshape(c_n.shape)
    # replace zero by coefficient at proper position
    for i in range(tmp.shape[0]):
        for j in range(tmp.shape[1]):
            for val in tmp[i, j]["idx"]:
                fit_n[val, i, j] = c_n[val, i, j]
    # do inverse Fourier transformation to get fit
    fourier_fit = np.fft.irfft(fit_n, corr.shape[0], axis=0)
    r2_val = 1 - (
        1
        - r2_score(
            corr.values.reshape(corr.shape[0], -1),
            fourier_fit.reshape(corr.shape[0], -1),
            multioutput="raw_values",
        ).reshape(corr.shape[1:])
    ) * (corr.shape[0] - 1) / (corr.shape[0] - np.count_nonzero(fit_n, axis=0) - 1)
    # the return period can be calculated via f=Nt/idx because the inverse
    # Fourier transform is given by
    # corr[n] = c_0 + \sum_{idx} [
    # 2*Re(c_{idx})*cos(2*pi*idx*n/Nt) - 2*Im(c_{idx})*sin(2*pi*idx*n/Nt) ]
    frequencies = np.fft.rfftfreq(NT)
    res[r2_val >= R2_THRESHOLD] = np.array(
        [
            1.0 / frequencies[val["idx"][0]]
            for val in tmp_non_zero[r2_val >= R2_THRESHOLD]
        ]
    )
    # check if corr is not constant
    res[np.sum(np.abs(c_n[1:]), axis=0) < 1e-10] = 1
    # make sure that gcd is also in list of the largest indices
    res[
        np.array(
            [
                gcd[i, j] not in tmp[i, j]["idx"]
                for i in range(tmp.shape[0])
                for j in range(tmp.shape[1])
            ],
        ).reshape(tmp.shape)
    ] = None
    # check if corr is significant
    res[(np.max(np.abs(corr), axis=0) < EPS_CORR).values] = None
    # check if dominant period is significant (cmp. to red noise)
    res[~significance_mask] = None
    return xr.DataArray(
        data=res,
        dims=corr.dims[1:],
        coords=corr.drop_vars("time").coords,
        attrs=corr.attrs,
    )


def _max_idx_val(
    data: Union[xr.Dataset, xr.DataArray], event: str, event1: str, no_trend: bool
) -> xr.DataArray:
    """Computes max value and index position of time correlation function"""
    # calculate time correlation
    if isinstance(data, xr.Dataset):
        corr = _calc_time_corr_array(data[event], data[event1])
    else:
        corr = _calc_time_corr(data, event, event1)
    if no_trend:
        # subtract linear trend (only slope-contribution)
        trend = (
            corr.polyfit("time", 1).sel(degree=1)["polyfit_coefficients"] * corr.time
        ).transpose(*["time", "lat", "lon"])
        corr = corr - trend
    # return significant return period
    return _determine_dominant_return_period_array(corr)


# pylint: disable=too-many-instance-attributes
class TimeAnalysisImpacts:
    """This class contains all data and methods to perform the time series analysis"""

    # pylint: disable=dangerous-default-value
    def __init__(
        self,
        log: logging,
        ssp: list,
        impact_type: str,
        use_all_gcms: bool = USE_ALL_GCM_MODELS,
        use_all_mods: bool = USE_ALL_IMP_MODELS,
        use_mean: bool = USE_MODEL_MEAN,
    ):
        self.use_all_gcms = use_all_gcms
        self.use_all_mods = use_all_mods
        self.log = log
        self.dominant_return_period_t0 = None
        self.dominant_return_period_t0_no_trend = None
        self.local_dominant_return_period = None
        self.local_dominant_return_period_no_trend = None
        self.use_model_mean = use_mean
        self.data_path = INPUT_DATA_PATH if not self.use_model_mean else DATA_MEAN_PATH
        self.impact_type = impact_type
        if self.use_all_gcms:
            self.climate_model = ALL_GCM_MODELS[self.impact_type]
        elif self.use_model_mean:
            self.climate_model = ["model-mean"]
        else:
            self.climate_model = [SINGLE_GCM_MODEL[self.impact_type]]
        if self.use_all_mods:
            self.impact_model = ALL_IMPACT_MODELS[self.impact_type]
        elif self.use_model_mean:
            self.impact_model = ["model-mean"]
        else:
            self.impact_model = [SINGLE_IMPACT_MODEL[self.impact_type]]
        self.ssp = ssp
        self.ssp_name = None
        self.get_ssp_scenario_name()
        # parse impact time series
        self.impact_time_series = None
        self.read_data()
        if self.impact_time_series is not None:
            # calculate event impacts in each time bin
            self.impact_count_t0 = {
                key: {} for key in self.impact_time_series.data_vars.keys()
            }
            self.total_count = {
                key: {} for key in self.impact_time_series.data_vars.keys()
            }

    def get_ssp_scenario_name(
        self,
    ) -> None:
        """
        Determines SSP scenario name based on `self.ssp`
        """
        if self.ssp == ["picontrol"]:
            self.ssp_name = "picontrol"
            return
        if (
            len(self.ssp) == 2
            and "historical" in self.ssp
            and "picontrol" not in self.ssp
        ):
            self.ssp_name = self.ssp.copy()
            self.ssp_name.remove("historical")
            if self.ssp_name[0].startswith("ssp"):
                self.ssp_name = self.ssp_name[0]
                return
        if self.ssp == ["historical"]:
            self.ssp_name = "historical"
            return
        raise NameError(f"Cannot parse ssp_name (folder name) from {self.ssp}")

    # pylint: disable=too-many-branches, too-many-statements
    def read_data(
        self,
    ) -> None:
        """Read impacts data and store internally"""
        extreme_dict = {
            name + "_" + gcm: []
            for name in self.impact_model
            for gcm in self.climate_model
        }
        # parse files in DATA_PATH and store in data frame
        filenames = glob.glob(
            os.path.join(self.data_path, "*.nc4")
            if self.use_model_mean
            else os.path.join(self.data_path, "**/*.nc*"),
            recursive=True,
        )
        for filename in filenames:
            if not (filename.endswith(".nc4") or filename.endswith(".nc")):
                self.log.error(
                    "File name extension is expected to be .nc4 or .nc"
                    f"but got {filename.split('.')[-1]} instead"
                )
                raise ValueError
            if not self.use_model_mean:
                # make sure that subpaths match impact type
                if (
                    os.path.relpath(filename, self.data_path).split(os.sep)[0]
                    != self.impact_type
                ):
                    continue
                # make sure that subpaths match impact model
                if (
                    os.path.relpath(filename, self.data_path).split(os.sep)[1]
                    not in self.impact_model
                ):
                    continue
            # remove filetype ending and split
            _filename = os.path.basename(filename).split(".")[0].split("_")
            # check whether data corresponds to desired data
            if (
                _filename[3] != self.impact_type
                or _filename[0] not in self.impact_model
                or _filename[1] not in self.climate_model
                or _filename[2] not in self.ssp
                or _filename[6] != "landarea"
            ):
                continue
            self.log.info(f"Reading file {filename}...")
            # append data into a dataframe for each extreme event
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                extreme_ds = xr.open_dataset(
                    filename,
                    decode_times=_filename[3]
                    not in [
                        "tropicalcyclonedarea",
                        "burntarea",
                        "cropfailedarea",
                        "driedarea",
                        "floodedarea",
                    ]
                    or _filename[3] == "burntarea"
                    and _filename[2] == "picontrol"
                    and _filename[0] == "classic"
                    and _filename[1] == "gfdl-esm4"
                    and _filename[7] == "1601",
                )
            # TODO: remove workaround once cama-flood file is fixed (first year appears twice)
            if (
                _filename[3] == "floodedarea"
                and _filename[0] == "h08"
                and _filename[1] == "ukesm1-0-ll"
                and _filename[2] == "picontrol"
            ):
                extreme_ds = extreme_ds.isel(time=slice(1, None))
            if "dt" in extreme_ds.time.attrs:
                extreme_ds["time"] = extreme_ds["time"].dt.year
            if "time_bnds" in extreme_ds.variables:
                extreme_ds = extreme_ds.drop_vars("time_bnds")
            if "depth" in extreme_ds.dims:
                if len(extreme_ds.depth) == 1:
                    extreme_ds = extreme_ds.squeeze("depth")
                else:
                    raise NotImplementedError("depth coordinate cannot be collapsed!")
            # extract year from different time formats if applicable:
            if not self.use_model_mean:
                if extreme_ds.time.dtype in [int, 'int64']:
                    pass
                elif not hasattr(extreme_ds.time, "units"):
                    # keep only year from time index
                    extreme_ds["time"] = extreme_ds.time.dt.year
                elif extreme_ds.time.units.startswith("years since"):
                    extreme_ds["time"] = (
                        extreme_ds.time
                        + int(
                            extreme_ds.time.units.replace("years since ", "").split(
                                "-"
                            )[0]
                        )
                    ).astype(int)
                elif extreme_ds.time.units.startswith("months since"):
                    extreme_ds["time"] = (
                        extreme_ds.time / 12
                        + int(
                            extreme_ds.time.units.replace("months since ", "").split(
                                "-"
                            )[0]
                        )
                    ).astype(int)
                elif (
                    extreme_ds.time.units.startswith("days since")
                    and extreme_ds.time.calendar == "365_day"
                ):
                    extreme_ds["time"] = (
                        extreme_ds.time / 365
                        + int(
                            extreme_ds.time.units.replace("days since ", "").split("-")[
                                0
                            ]
                        )
                    ).astype(int)
                extreme_ds = extreme_ds.rename(
                    {"exposure": _filename[0] + "_" + _filename[1]}
                )
                extreme_dict[_filename[0] + "_" + _filename[1]].append(extreme_ds)
            else:
                extreme_dict[_filename[0] + "_" + self.climate_model[0]].append(
                    extreme_ds.rename(
                        {_filename[3]: _filename[3] + "_" + self.climate_model[0]}
                    )
                )
            self.log.info(f"{filename} successfully parsed!\n")
        # get rid of empty data
        extreme_dict = {key: val for key, val in extreme_dict.items() if len(val)}
        # concat data into a single large dataframe
        self.log.info("Merging data into a single dataframe...")
        self.impact_time_series = [
            xr.concat(val, dim="time") for val in extreme_dict.values()
        ]
        # reduce time range to desired one
        self.impact_time_series = [
            df.sel(time=range(min(t_0s), max(t_0s) + 2 * NT + 1))
            for df in self.impact_time_series
        ]
        # check whether data is complete
        count = {
            "lon": 360 * 2,
            "lat": 180 * 2,
            "time": (max(t_0s) + 2 * NT + 1 - min(t_0s)),
        }
        if any(count != df.sizes for df in self.impact_time_series):
            self.log.error("Data is incomplete. Check input files!")
            raise FileNotFoundError
        # merge all columns into a single data frame
        self.impact_time_series = xr.merge(self.impact_time_series).fillna(0)
        self.log.info("Merge done!")
        # final checks
        if (
            len(self.impact_time_series.data_vars.keys()) != len(extreme_dict.keys())
            or count != self.impact_time_series.sizes
        ):
            raise AssertionError(
                "ISIMIP xarray does not have correct size, "
                f"check input files: {self.impact_time_series.sizes} != {count}\n"
                f"{self.impact_time_series}"
            )

    def count_impacts(
        self,
    ) -> None:
        """This function counts impacts in `DT` bins and stores them"""
        nan_data = np.full(
            (
                self.impact_time_series.dims["lat"],
                self.impact_time_series.dims["lon"],
            ),
            np.nan,
        )
        impacted_area = self.impact_time_series * surface_area
        if self.impact_type == "burntarea":
            # TODO: make sure that we really use the 1% cap
            # rescale by 100*100 due to the cap at 1% in burnt area
            impacted_area = impacted_area / 10.000
        for impact_event in self.impact_count_t0.keys():
            self.log.info(f"Counting total {impact_event} affected area...")
            self.impact_count_t0[impact_event] = {t_0: nan_data for t_0 in t_0s}
            for t_0 in t_0s:
                # determine limits of time window, namely [t,t+2*Nt)
                t_start = t_0
                t_final = t_0 + NT * 2
                # retrieve indices for relevant times in data frames
                val_ds = self.impact_time_series[impact_event].sel(
                    time=range(t_start, t_final)
                )
                # count extreme events in time bins
                self.impact_count_t0[impact_event][t_0] = xr.DataArray(
                    data=np.count_nonzero(val_ds, axis=val_ds.get_axis_num("time")),
                    coords={
                        "lat": val_ds.coords["lat"],
                        "lon": val_ds.coords["lon"],
                    },
                    name=self.impact_type,
                    attrs={
                        "standard_name": f"{impact_event} counts in time bin [{t_start},{t_final})",
                        "unit": "1",
                        "scenario": self.ssp_name,
                        "impact_model": impact_event.split("_")[0],
                        "climate_model": impact_event.split("_")[1],
                    },
                )
            self.total_count[impact_event] = {
                t_0: None for t_0 in range(min(t_0s), max(t_0s) + 1)
            }
            # count worldwide extreme event for all years along t_0s range
            for t_0 in range(min(t_0s), max(t_0s) + 1):
                # determine limits of time window, namely [t,t+2*Nt)
                t_start = t_0
                t_final = t_0 + NT * 2
                # retrieve indices for relevant times in data frames
                val_ds = impacted_area[impact_event].sel(time=range(t_start, t_final))
                # calculate total affected area (affected area share * cell area)
                self.total_count[impact_event][t_0] = val_ds.sum()
            self.log.info(
                f"Total {impact_event} affected area successfully calculated!"
            )
        # check if all nan values where replaced
        for impact_event in self.impact_count_t0.values():
            for impact_count in impact_event.values():
                if np.any(np.isnan(impact_count.values)):
                    raise RuntimeError("Counting did not account for all coordinates!")

    def calculate_dominant_return_period(self, no_trend: bool):
        """Calculates dominant return period for all events"""
        # initialize result object
        if no_trend:
            self.dominant_return_period_t0_no_trend = {}
        else:
            self.dominant_return_period_t0 = {}
        container = (
            self.dominant_return_period_t0
            if not no_trend
            else self.dominant_return_period_t0_no_trend
        )
        # correlation function is evaluated for a time window of Nt
        for event in self.impact_time_series.data_vars.keys():
            self.log.info(f"Calculating return period for {event}-{event}...")
            container[(event, event)] = {}
            for t_0 in t_0s:
                # determine limits of time window
                t_start = t_0
                t_final = t_0 + NT * 2
                # retrieve data for relevant times in data frames
                val_ds = self.impact_time_series.sel(time=range(t_start, t_final))
                # group by location and determine dominant return period
                dominant_freq = _max_idx_val(
                    val_ds[list({event, event})], event, event, no_trend=no_trend
                )
                container[(event, event)][t_0] = dominant_freq
                container[(event, event)][t_0].attrs["standard_name"] = (
                    (
                        f"{self.impact_type}-{event} dominant return period "
                        f"in time bin [{t_start},{t_final})",
                    )
                    if not no_trend
                    else (
                        f"{self.impact_type}-{event} dominant return period "
                        f"without trend in time bin [{t_start},{t_final})",
                    )
                )
                container[(event, event)][t_0].attrs["unit"] = "1"
                container[(event, event)][t_0].attrs["scenario"] = self.ssp_name
                container[(event, event)][t_0].attrs["impact_model"] = (
                    event.split("_")[0],
                    event.split("_")[0],
                )
                container[(event, event)][t_0].attrs["climate_model"] = (
                    event.split("_")[1],
                    event.split("_")[1],
                )
            self.log.info(
                f"Return period for {self.impact_type}-{event} successfully calculated!"
            )

    def calculate_local_dominant_return_period(self, no_trend: bool) -> None:
        """Calculate LOCAL dominant return periods"""
        if not no_trend:
            self.local_dominant_return_period = {}
            container = self.local_dominant_return_period
        else:
            self.local_dominant_return_period_no_trend = {}
            container = self.local_dominant_return_period_no_trend
        # check time dependence of correlation close to location
        # pylint: disable=too-many-locals, too-many-nested-blocks
        for location, (lat, long) in LOCATIONS.items():
            extreme_ds_local = self.impact_time_series.sel({"lon": long, "lat": lat})
            container[location] = {}
            # select extreme event
            for event in extreme_ds_local.data_vars.keys():
                # correlation function is evaluated for a time window of Nt
                container[location][(event, event)] = {}
                for t_0 in t_0s:
                    # determine limits of time window, namely [t,t+Nt)
                    # for f_i and [t+n,t+Nt+n) for f_j, where n=0,..,Nt-1
                    t_start = t_0
                    t_final = t_0 + NT * 2
                    # retrieve indices for relevant times in data frames
                    val_ds = extreme_ds_local.sel(time=range(t_start, t_final))
                    # compute correlation function for shift in [0,Nt]
                    corr = _calc_time_corr(val_ds, event, event)
                    if no_trend:
                        # subtract linear trend (only slope-contribution)
                        corr = corr - np.polyfit(np.arange(NT), corr, 1)[0] * np.arange(
                            NT
                        )
                    # compute dominant frequency
                    res = _determine_dominant_return_period(corr)
                    container[location][(event, event)][t_0] = {
                        "corr": corr,
                        "r2_val": res[0],
                        "max_fit": res[1],
                        "dominant_ret_per": res[2],
                        "c_n": res[3],
                        "sorted_idx": res[4],
                    }

    def store_extreme_count_bins(self) -> None:
        """stores extreme event counts within time bins in output file"""
        for impact_type, data in self.impact_count_t0.items():
            self.log.info(
                f"Storing {self.impact_type}-{impact_type} event counting to file..."
            )
            full_path = os.path.join(
                OUTPUT_PATH,
                "event_counts",
                self.impact_type,
                impact_type.split("_")[0],
            )
            if not os.path.exists(full_path):
                os.makedirs(full_path)
            # store impact count data as netcdf
            xr.Dataset({str(key): val for key, val in data.items()}).to_netcdf(
                os.path.join(
                    full_path,
                    f"{impact_type.split('_')[0]}_{self.ssp_name}"
                    f"_{self.impact_type}_{impact_type.split('_')[1]}"
                    f"_Nt{NT}_extreme_event_counts.nc",
                )
            )
            self.log.info(
                f"{self.impact_type}-{impact_type} event counting successfully stored to file!"
            )
            self.log.info(
                f"Storing total {self.impact_type}-{impact_type} event counting to file..."
            )
            # store impact count data as netcdf
            with open(
                os.path.join(
                    full_path,
                    f"{impact_type.split('_')[0]}_{self.ssp_name}"
                    f"_{self.impact_type}_{impact_type.split('_')[1]}"
                    f"_Nt{NT}_total_extreme_event_counts.csv",
                ),
                "w",
                encoding="utf-8",
                newline="",
            ) as file:
                fields = ["year", "counts"]
                rows = np.array(list(self.total_count[impact_type].items()))
                writer = csv.writer(file)
                writer.writerow(fields)
                writer.writerows(rows)

            self.log.info(
                f"{self.impact_type}-{impact_type} "
                "total event counting successfully stored to file!"
            )

    def store_dominant_return_period(self, no_trend: bool):
        """Stores dominant return period (with or without detrending) as netcdf"""
        if no_trend:
            container = self.dominant_return_period_t0_no_trend
            output_subdir = "detrended"
        else:
            container = self.dominant_return_period_t0
            output_subdir = "original"
        for impact_type, data in container.items():
            self.log.info(
                f"Storing {self.impact_type}-{impact_type} event dominant return period to file..."
            )
            full_path = os.path.join(
                OUTPUT_PATH,
                output_subdir,
                "dominant_return_period",
                f"{self.impact_type}_{self.impact_type}",
                f"{impact_type[0].split('_')[0]}_" f"{impact_type[1].split('_')[0]}",
            )
            if not os.path.exists(full_path):
                os.makedirs(full_path)
            # store impact count data as netcdf
            xr.Dataset({str(key): val for key, val in data.items()}).to_netcdf(
                os.path.join(
                    full_path,
                    f"{self.impact_type}_{impact_type[0].split('_')[1]}"
                    f"_{self.impact_type}_{impact_type[1].split('_')[1]}"
                    f"_{self.ssp_name}_extreme_event_Nt{NT}_dominant_frequency.nc",
                )
            )
            self.log.info(
                f"{self.impact_type}-{impact_type} "
                "event dominant return period successfully stored to file!"
            )

    def store_average_impact_probability(self) -> None:
        """Stores average impact probability in file"""
        if self.use_all_mods and self.use_all_gcms:
            self.log.info("Storing impact probabilities and affected counts...")
            path_name = os.path.join(OUTPUT_PATH, "statistical_test")
            if not os.path.exists(path_name):
                os.makedirs(path_name)
            # calculate probabilities from impacted areas / all areas
            probabilities = (self.impact_time_series > 0).sum(["time"]) / (
                self.impact_time_series.count(["time"])
            )
            probabilities.to_netcdf(
                os.path.join(
                    path_name,
                    f"{self.impact_type}_{self.ssp_name}_NT{NT}_NT0{len(t_0s)}"
                    "_impact_probability.nc",
                )
            )
            # count number of non-trivial time series
            affected_counts = np.sum(
                [
                    self.impact_time_series.sel(time=range(t0, t0 + NT))
                    .sum(dim="time")
                    .to_array()
                    > 0
                    for t0 in t_0s
                ]
            )
            with open(
                os.path.join(
                    path_name,
                    "total_affected_counts"
                    + f"_{self.impact_type}_{self.ssp_name}_NT{NT}_NT0{len(t_0s)}.csv",
                ),
                "w",
                encoding="utf-8",
            ) as csv_file:
                writer = csv.writer(csv_file)
                writer.writerow([self.impact_type, affected_counts])
            self.log.info(
                "Impact probabilities and affected counts successfully stored!"
            )


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
    return parser.parse_args()


def main():
    """Main function that reads data and calculates convolution"""
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
        # create log directory if it doesn't exist
        log_dir = os.path.join(os.getcwd(), LOG_PATH)
        os.makedirs(log_dir, exist_ok=True)
        # include log file
        log_handlers.append(
            logging.FileHandler(os.path.join(log_dir, log_filename))
        )
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(message)s",
        level=logging.INFO,
        handlers=log_handlers,
    )
    logging.info("Starting execution of time series analysis")
    logging.info("==========================================")
    # pylint:disable=logging-fstring-interpolation
    logging.info(f"SSP={flags.ssp_scenario}")
    logging.info(f"Impact type={flags.impact_type}")
    logging.info(f"NT={NT}")
    logging.info(f"Reference times={t_0s}")
    logging.info(f"Threshold for correlation={EPS_CORR}")
    logging.info(f"Threshold for R^2={R2_THRESHOLD}")
    logging.info("==========================================")
    if USE_MODEL_MEAN and USE_ALL_GCM_MODELS:
        raise ValueError(
            "Code cannot operate with USE_MODEL_MEAN and USE_ALL_GCM_MODELS simultaneously!"
        )
    # initialize time series analysis object
    time_series_analysis = TimeAnalysisImpacts(
        impact_type=flags.impact_type,
        use_all_gcms=USE_ALL_GCM_MODELS,
        use_all_mods=USE_ALL_IMP_MODELS,
        use_mean=USE_MODEL_MEAN,
        ssp=ALL_SSP_SCENARIOS[flags.ssp_scenario],
        log=logging,
    )
    if time_series_analysis.impact_time_series is None:
        return
    time_series_analysis.store_average_impact_probability()
    if time_series_analysis.impact_time_series is not None:
        time_series_analysis.count_impacts()
    # store them extreme event counts in time bins
    time_series_analysis.store_extreme_count_bins()
    # calculate detrended results only for non-picontrol
    no_trends = [True, False] if flags.ssp_scenario != "picontrol" else [False]
    for no_trend in no_trends:
        logging.info(f"Analysing detrended results: {no_trend}...")
        if time_series_analysis.impact_time_series is not None:
            # calculate local dominant return periods
            time_series_analysis.calculate_local_dominant_return_period(
                no_trend=no_trend
            )
        if RUN_DOMINANT_FREQUENCY_CALC:
            # calculate dominant return period
            time_series_analysis.calculate_dominant_return_period(no_trend=no_trend)
    # store dominant return period
    if time_series_analysis.dominant_return_period_t0_no_trend is not None:
        time_series_analysis.store_dominant_return_period(no_trend=True)
    if time_series_analysis.dominant_return_period_t0 is not None:
        time_series_analysis.store_dominant_return_period(no_trend=False)
    end = datetime.now(pytz.timezone("UTC"))
    logging.info(
        "Thank you for running the simulation! \n Total time: %s", str(end - start)
    )


# Press the green button in the gutter to run the script.
if __name__ == "__main__":
    main()
