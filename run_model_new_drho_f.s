#!/bin/bash

source ~/miniconda3/bin/activate pss-env

#set -x

sh_dir="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

src_dir="$(realpath "${sh_dir}/src")"

data_dir="$(realpath "${sh_dir}/data")"

if (( $# != 5 )); then
    echo "Usage: run_model.s model_name rgamma alpha chi rhoseed"
    exit 1
fi


model=$1
rgamma=$(python3 -c "print('{:.2f}'.format($2))")
alpha=$(python3 -c "print('{:.2f}'.format($3))")
chi=$(python3 -c "print('{:.2f}'.format($4))")
rhoseed=$(python3 -c "print('{:.2f}'.format($5))")

run=2
gamma0=200.0
gammaxx=1.0
p0=1.0
gammayy=$(python3 -c "print('{:.2f}'.format($gammaxx*$rgamma))")
D=150
a=30
b=90
d=50
K=5
T=50
n_steps=5e+5
dt_dump=0.1
lambda=5
rho_in=3.2
rhoisoend=3.75
rhonemend=10.0
mx=100
my=100
dx=1.0
dy=1.0

save_dir="${sh_dir}/data/$model/gamma0_${gamma0}_rhoseed_${rhoseed}/rgamma_${rgamma}_alpha_${alpha}_chi_${chi}/run_${run}"

if [ ! -d $save_dir ]; then
    mkdir -p $save_dir
fi

params_file="${save_dir}/parameters.json"

echo \
"
{
    "\"run\"" : $run,
    "\"T\"" : $T,
    "\"n_steps\"" : $n_steps,
    "\"dt_dump\"" : $dt_dump,
    "\"K\"" : $K,
    "\"Gamma0\"" : $gamma0,
    "\"alpha\"" : $alpha,
    "\"gammayy\"" : $gammayy,
    "\"gammaxx\"" : $gammaxx,
    "\"chi\"": $chi,
    "\"lambda\"": $lambda,
    "\"a\"": $a,
    "\"b\"": $b,
    "\"d\"": $d,
    "\"D\"": $D,
    "\"p0\"": $p0,
    "\"rhoseed\"" : $rhoseed,
    "\"rho_in\"" : $rho_in,
    "\"rhoisoend\"" : $rhoisoend,
    "\"rhonemend\"" : $rhonemend,
    "\"mx\"" : $mx,
    "\"my\"" : $my,
    "\"dx\"" : $dx,
    "\"dy\"" : $dy
}
" > $params_file

python3 -m models.$model -s $save_dir
python3 -m src.analysis.create_avgs -s $save_dir
python3 -m src.analysis.create_videos -s $save_dir