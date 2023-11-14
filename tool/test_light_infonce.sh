export PYTHONPATH=./
eval "$(conda shell.bash hook)"
PYTHON=python

TEST_CODE=test_light_infonce.py

dataset=$1
exp_name=$2
exp_dir=exp/${dataset}/${exp_name}
model_dir=${exp_dir}/model
result_dir=${exp_dir}/result
config=config/${dataset}/${dataset}_${exp_name}.yaml

mkdir -p ${result_dir}/last
mkdir -p ${result_dir}/best

now=$(date +"%Y%m%d_%H%M%S")
cp ${config} tool/test.sh tool/${TEST_CODE} ${exp_dir}

now=$(date +"%Y%m%d_%H%M%S")
$PYTHON ${exp_dir}/${TEST_CODE} \
  --config=${config} \
  save_path ${exp_dir}