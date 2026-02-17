"""
This file contains all settings needed for the extreme event analysis
"""
import os

# use percentile definition for all impacts
USE_PERCENTILE = False
# use F-statistic for significance filtering
USE_CHI2_FILTER = True
# use AR model to determine dominant period
USE_AR_MODEL = True
# use ISIMIP3a data
USE_ISIMIP3A = False
# use specific ISIMIP3b to ISIMIP3a comparison output directory
USE_ISIMIP3B_TO_3A_COMPARISON = False
# use (input) model mean instead of single model combination or all models
USE_MODEL_MEAN = False
# use (output) result median instead of single model combination or all models
USE_RESULT_MEDIAN = True
# run on PIK cluster
USE_PIK_CLUSTER = False
# run with all gcm models
USE_ALL_GCM_MODELS = False
# run with all impact models
USE_ALL_IMP_MODELS = False
# run dominant frequency calculations
RUN_DOMINANT_FREQUENCY_CALC = True
# run crop resolved dominant return period
RUN_CROP_TYPE_RESOLVED = False
# run only regional analysis
RUN_REGIONAL_ANALYSIS = False
# time window for observing time correlations, namely N_t
NT = 25
# reference times
t_0s = [1850, 1900, 1950, 2000, 2050]
# time correlation tolerance (i.e. minimal value to account for)
EPS_CORR = 0.0001
# R^2 threshold for accepting significant frequency
R2_THRESHOLD = 0.5
# locations of interest
LOCATIONS = {
    "Potsdam": (52.25, 13.25),  # (lat, lon)
    "Brisbane": (-27.75, 153.25),
    "Ahmedabad": (23.25, 72.25),
    "Madhubani": (25.75, 85.75),
    "Lusaka": (-15.25, 28.25),
}
# selection of single GCM model
SINGLE_GCM_MODEL = (
    {
        "cropfailedarea": "mpi-esm1-2-hr",  # crop-failure
        "heatwavedarea": "gfdl-esm4",  # heatwave
        "burntarea": "gfdl-esm4",  # wildfire
    }
    if not USE_ISIMIP3A
    else {
        "cropfailedarea": "gswp3-w5e5",
        "burntarea": "gswp3-w5e5",
        "heatwavedarea": "gswp3-w5e5",
    }
)
SINGLE_IMPACT_MODEL = {
    "cropfailedarea": "EPIC-IIASA",  # crop-failure
    "heatwavedarea": "hwmid-humidex" if not USE_PERCENTILE else "hwmid-none",  # heatwave
    "burntarea": "lpjml5-7-10-fire",  # wildfire
}

#########################################
# input data path (all model combinations)
if not USE_PIK_CLUSTER:
    INPUT_DATA_PATH = os.path.join(os.getcwd(), "data", "yearly_resolution")
    REGION_INPUT_DATA_PATH = os.path.join(os.getcwd(), "data", "out")
elif not USE_ISIMIP3A:
    INPUT_DATA_PATH = os.path.join(
        "/",
        "p",
        "tmp",
        "karimza",
        "extreme-climate-impacts-exposure-isimip3",
        "isimip3",
        "data",
        "in",
    ) if not USE_PERCENTILE else os.path.join(
        "/",
        "p",
        "tmp",
        "karimza",
        "extreme-climate-impacts-exposure-isimip3b-all-percentile",
        "data",
        "in",
    )
    REGION_INPUT_DATA_PATH = os.path.join(*INPUT_DATA_PATH.split(os.sep)[:-1], "out")
else:
    INPUT_DATA_PATH = os.path.join(
        "/",
        "p",
        "tmp",
        "karimza",
        "extreme-climate-impacts-exposure-isimip-3-a",
        "isimip3",
        "data",
        "in",
    ) if not USE_PERCENTILE else os.path.join(
        "/",
        "p",
        "tmp",
        "karimza",
        "extreme-climate-impacts-exposure-isimip3a-all-percentile",
        "data",
        "in",
    )
    REGION_INPUT_DATA_PATH = None

# model mean data path
DATA_MEAN_PATH = os.path.join(os.getcwd(), "data", "yearly_resolution", "model-mean")


# output path
def get_output_path(use_isimip3a, use_isimip3b_to_3a_comp, run_croptype, use_percentile) -> os.path:
    """Determines output path"""
    if not use_isimip3a and not use_isimip3b_to_3a_comp and not run_croptype:
        output_path = os.path.join(os.getcwd(), "output") if not use_percentile else os.path.join(
            os.path.join(os.getcwd(), "output_percentile_definition")
        )
    elif use_isimip3a and not use_isimip3b_to_3a_comp and not run_croptype:
        output_path = os.path.join(os.getcwd(), "output_isimip3a") if not use_percentile else os.path.join(
            os.path.join(os.getcwd(), "output_isimip3a_percentile_definition")
        )
    elif use_isimip3b_to_3a_comp and not use_isimip3a and not run_croptype:
        output_path = os.path.join(os.getcwd(), "output_isimip3b_comp") if not use_percentile else os.path.join(
            os.path.join(os.getcwd(), "output_isimip3b_comp_percentile_definition")
        )
    elif run_croptype:
        output_path = os.path.join(os.getcwd(), "output_cropresolved") if not use_percentile else os.path.join(
            os.path.join(os.getcwd(), "output_cropresolved_percentile_definition")
        )
    return output_path


OUTPUT_PATH = get_output_path(USE_ISIMIP3A, USE_ISIMIP3B_TO_3A_COMPARISON, RUN_CROP_TYPE_RESOLVED, USE_PERCENTILE)
REGION_OUTPUT_PATH = OUTPUT_PATH.replace("output_", "output_regional_")
# output path for tests
TEST_OUTPUT_PATH = os.path.join(os.getcwd(), "tests", "output")
# plots path
if not USE_ISIMIP3A:
    PLOTS_PATH = os.path.join(os.getcwd(), "plots") if not USE_PERCENTILE else os.path.join(
        os.getcwd(), "plots_percentile_definition"
    )
else:
    PLOTS_PATH = os.path.join(os.getcwd(), "plots_isimip3a") if not USE_PERCENTILE else os.path.join(
        os.getcwd(), "plots_isimip3a_percentile_definition"
    )
# name of the logging directory
LOG_PATH = "Logs"
# gridded surface area path
SURFACE_AREA_PATH = os.path.join(os.getcwd(), "data", "ISIMIP_grid_area.nc")
# Southern Oscillation index path
# Citation:
# SOI Data provided by the Climate Analysis Section, NCAR, Boulder, USA, Trenberth (1984).
# Updated regularly. Accessed 14 July 2025.
SOI_DATA_PATH = os.path.join(os.getcwd(), "data", "SOI_data.nc")
#############################
# all available ssp scenarios
ALL_SSP_SCENARIOS = (
    {
        "picontrol": ["picontrol"],
        "ssp585": ["ssp585", "historical"],
        "ssp126": ["ssp126", "historical"],
        "ssp370": ["ssp370", "historical"],
        "historical": ["historical"],
    }
    if not USE_ISIMIP3A
    else {"picontrol": ["picontrol"], "historical": ["historical"]}
)
# all impact model
ALL_IMPACT_MODELS = (
    {
        "cropfailedarea": [
            "CROVER",
            "CYGMA1p74",
            "EPIC-IIASA",
            "ISAM",
            "LDNDC",
            "LPJmL",
            "PEPIC",
            "PROMET",
        ],  # crop-failure
        "heatwavedarea": [
            "hwmid-humidex" if not USE_PERCENTILE else "hwmid-none",
        ],  # heatwave
        "burntarea": ["classic", "visit", "lpjml5-7-10-fire"],  # wildfire
    }
    if not USE_ISIMIP3A
    else {
        "cropfailedarea": [
            "ACEA",
            "CROVER",
            "CYGMA1p74",
            "EPIC-IIASA",
            "ISAM",
            "LDNDC",
            "LPJmL",
            "pDSSAT",
            "PEPIC",
        ],  # crop-failure
        "heatwavedarea": [
            "hwmid-humidex" if not USE_PERCENTILE else "hwmid-none",
        ],  # heatwave
        "burntarea": ["classic", "lpjml5-7-10-fire", "visit"],  # wildfire
    }
)
# all GCM models
ALL_GCM_MODELS = (
    {
        "cropfailedarea": [
            "gfdl-esm4",
            "ukesm1-0-ll",
            "ipsl-cm6a-lr",
            "mpi-esm1-2-hr",
            "mri-esm2-0",
        ],
        "heatwavedarea": [
            "gfdl-esm4",
            "ukesm1-0-ll",
            "ipsl-cm6a-lr",
            "mpi-esm1-2-hr",
            "mri-esm2-0",
        ],
        "burntarea": [
            "gfdl-esm4",
            "ukesm1-0-ll",
            "ipsl-cm6a-lr",
            "mpi-esm1-2-hr",
            "mri-esm2-0",
        ],
    }
    if not USE_ISIMIP3A
    else {
        "cropfailedarea": ["gswp3-w5e5"],
        "heatwavedarea": [
            "20crv3",
            "20crv3-era5",
            "20crv3-w5e5",
            "gswp3-w5e5",
        ],
        "burntarea": [
            "gswp3-w5e5",
        ],
    }
)
