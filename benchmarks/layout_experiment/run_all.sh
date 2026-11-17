#!/bin/zsh
# Layout experiment, end to end, on any Apple-silicon Mac with this checkout installed by
# install/macOS/install_firm3d_metal.sh (its .venv is used; nothing is installed).
#
#   benchmarks/layout_experiment/run_all.sh
#
# Builds the harness into bin/, writes the inputs into data/ and data_fb/ (about 0.5 GB,
# gitignored), runs the bitwise smoke tests and the timed sweeps (vacuum, small-n vacuum,
# finite beta), and writes summary-<host>.txt next to this script. Takes about 6 minutes on
# an M4 and keeps the GPU busy throughout; close GPU-heavy apps first. Timed rows append to
# each data directory's results.csv, so delete those files to start a clean sweep.
set -euo pipefail
HERE=${0:A:h}
REPO=${HERE:h:h}
PY=$REPO/.venv/bin/python
mkdir -p $HERE/bin

clang++ -std=c++17 -O2 -fobjc-arc $HERE/layout_bench.mm -framework Metal -framework Foundation \
    -o $HERE/bin/layout_bench

# the benchmark config names its field file relative to benchmarks/
(cd $REPO/benchmarks && $PY $HERE/prep_inputs.py $HERE/config.yaml && $PY $HERE/prep_inputs.py $HERE/config_fb.yaml)
mkdir -p $HERE/data_small
for f in table.f32 particles.f32 meta.txt; do ln -sf $HERE/data/$f $HERE/data_small/$f; done

for name in config config_small config_fb; do
    $PY $HERE/run_layout.py $HERE/$name.yaml smoke
    $PY $HERE/run_layout.py $HERE/$name.yaml timed
done

{
    echo "machine: $(sysctl -n machdep.cpu.brand_string), $(system_profiler SPDisplaysDataType | grep -m1 'Total Number of Cores' | sed 's/^ *//') GPU"
    echo "macOS: $(sw_vers -productVersion); firm3d_metal: $(git -C $REPO rev-parse --short HEAD)"
    for name in config config_small config_fb; do
        echo; echo "== $name"
        $PY $HERE/analyze.py $HERE/$name.yaml
    done
} | tee $HERE/summary-$(hostname -s).txt
