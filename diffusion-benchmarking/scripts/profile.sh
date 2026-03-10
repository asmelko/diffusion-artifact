#!/bin/bash

binary=$1
out_dir=$2

mkdir -p ${out_dir}

# Define the list of algorithms to test
algorithms=("biofvm" "lstcma" "lstcta" "lstcstai" "lstmtai" "lstmdtai" "lstmdtfai" "lstmfpabni")

# Write the initial problem JSON to a file
problem_file="${out_dir}/problem-$(date +%s).json"
cat > "$problem_file" <<EOF
{
    "dims": 3,
    "dx": 20,
    "dy": 20,
    "dz": 20,
    "nx": 100,
    "ny": 100,
    "nz": 100,
    "substrates_count": 1,
    "iterations": 50,
    "dt": 0.01,
    "diffusion_coefficients": 10000,
    "decay_rates": 0.01,
    "initial_conditions": 1000,
    "gaussian_pulse": false
}
EOF

# Write the initial params JSON to a file
params_file="${out_dir}/params-$(date +%s).json"
cat > "$params_file" <<EOF
{
    "benchmark_kind": "full_solve"
}
EOF

# Define the sets of values to test
n_values=($(seq 25 25 300))
substrates_values=(1 8 16 32 64)
counters=("PAPI_FP_OPS,PAPI_FP_INS" "PAPI_VEC_INS,PAPI_TOT_INS" "PAPI_L3_TCM,PAPI_L2_TCM,PAPI_L1_TCM" "PAPI_L2_DCM,PAPI_L1_DCM" "PAPI_BR_MSP")
params=("[1,1,1]" "[1,1,2]" "[1,1,4]" "[1,1,7]" "[1,1,14]" "[1,1,28]" "[1,1,56]" "[1,2,2]" "[1,2,4]" "[1,2,7]" "[1,2,14]" "[1,2,28]" "[1,2,56]" "[1,4,4]" "[1,4,7]" "[1,4,14]" "[1,4,28]" "[1,8,7]" "[1,8,14]")

for alg in "${algorithms[@]}"; do
    for n in "${n_values[@]}"; do
        for substrates in "${substrates_values[@]}"; do
            for param in "${params[@]}"; do
                for counter in "${counters[@]}"; do
                    # Update the JSON file in-place using Python
                    python3 -c "import json; f='$problem_file'; d=json.load(open(f)); d['nx']=${n}; d['ny']=${n}; d['nz']=${n}; d['substrates_count']=${substrates}; json.dump(d, open(f, 'w'), indent=4)"
                    python3 -c "import json; f='$param_file'; d=json.load(open(f)); d['cores_division']=${param}; json.dump(d, open(f, 'w'), indent=4)"

                    for dtype in "s" "d"; do
                        logdir="${out_dir}/profile_${alg}_${dtype}_${n}x${substrates}_${param}_${counter}"

                        if [ -d "${logdir}/papi_hl_output" ]; then
                            echo "Skipping $logdir (already exists)"
                            continue
                        fi

                        mkdir -p $logdir

                        export PAPI_EVENTS="${counter}"
                        export PAPI_OUTPUT_DIRECTORY="${logdir}"

                        echo "Running $alg $dtype with ${n}^3 @ ${substrates} and ${counter}"

                        if [ "$dtype" = "s" ]; then
                            ${binary} --problem "$problem_file" --params "$params_file" --alg "$alg" --profile
                        else
                            ${binary} --problem "$problem_file" --params "$params_file" --alg "$alg" --profile --double
                        fi
                    done
                done
            done
        done
    done
done
