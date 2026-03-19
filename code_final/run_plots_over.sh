#!/bin/bash

NT=25
T_0S="[1850, 1900, 1950, 2000, 2050]"
R2_THRESHOLD=0.5
EPS_CORR=0.0001
USE_ISIMIP3A="False"
USE_ISIMIP3B_TO_3A_COMPARISON="False"

USE_MODEL_MEAN="False"
USE_RESULT_MEDIAN="False"

USE_PIK_CLUSTER="False"
RUN_DOMINANT_FREQUENCY_CALC="True"
USE_ALL_GCM_MODELS="True"
USE_ALL_IMP_MODELS="True"

# change parameters in settings.py
sed -i "s/^NT = .*/NT = $NT/g" settings.py
sed -i "s/^t_0s = .*/t_0s = $T_0S/g" settings.py
sed -i "s/^R2_THRESHOLD = .*/R2_THRESHOLD = $R2_THRESHOLD/g" settings.py
sed -i "s/^EPS_CORR = .*/EPS_CORR = $EPS_CORR/g" settings.py
sed -i "s/^USE_MODEL_MEAN = .*/USE_MODEL_MEAN = $USE_MODEL_MEAN/g" settings.py
sed -i "s/^USE_ISIMIP3A = .*/USE_ISIMIP3A = $USE_ISIMIP3A/g" settings.py
sed -i "s/^USE_ISIMIP3B_TO_3A_COMPARISON = .*/USE_ISIMIP3B_TO_3A_COMPARISON = $USE_ISIMIP3B_TO_3A_COMPARISON/g" settings.py
sed -i "s/^USE_RESULT_MEDIAN = .*/USE_RESULT_MEDIAN = $USE_RESULT_MEDIAN/g" settings.py
sed -i "s/^USE_PIK_CLUSTER = .*/USE_PIK_CLUSTER = $USE_PIK_CLUSTER/g" settings.py
sed -i "s/^RUN_DOMINANT_FREQUENCY_CALC = .*/RUN_DOMINANT_FREQUENCY_CALC = $RUN_DOMINANT_FREQUENCY_CALC/g" settings.py
sed -i "s/^USE_ALL_GCM_MODELS = .*/USE_ALL_GCM_MODELS = $USE_ALL_GCM_MODELS/g" settings.py
sed -i "s/^USE_ALL_IMP_MODELS = .*/USE_ALL_IMP_MODELS = $USE_ALL_IMP_MODELS/g" settings.py


# available impact models and ssp scenarios
SSP_SCENARIOS="picontrol ssp585"
#IMPACT_TYPE="burntarea cropfailedarea heatwavedarea floodedarea"
IMPACT_TYPE=floodedarea

declare -A ISIMIP_IMPACT_LABEL=(
    ["cropfailedarea"]="a"
    ["heatwavedarea"]="b"
    ["burntarea"]="c"
    ["floodedarea"]="g"
    ["driedarea"]="h"
)
declare -A ISIMIP_IMPACT_PICONTROL_LABEL=(
    ["cropfailedarea"]="d"
    ["heatwavedarea"]="e"
    ["burntarea"]="f"
    ["floodedarea"]="i"
    ["driedarea"]="j"
)

for ssp in $SSP_SCENARIOS
do
  for type in $IMPACT_TYPE
  do
    echo ssp=$ssp impact_type=$type
    bash run_plots.sh $ssp $type
  done
done
