#!/bin/bash
# Install this FIRM3D fork, with its Metal GPU backend, on an Apple-Silicon Mac.
#
# Python packages go into .venv at the repository root (managed by uv, on uv's
# own Python); the system libraries the build links against (GSL, Open MPI)
# come from Homebrew. Conda is kept out of the build: run it with no conda
# environment active.
# Nothing outside the repository is installed without asking first. Safe to
# re-run: it rebuilds the C++/Metal extension and re-verifies every time, so
# it is also the way to update after a `git pull`.
#
# Usage, from anywhere inside the clone:
#
#     bash install/macOS/install_firm3d_metal.sh

set -euo pipefail

PYTHON_VERSION=3.12
BREW_PACKAGES=(gsl open-mpi)

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"

say() { printf '\n==> %s\n' "$1"; }
die() {
    printf '\nError: %s\n' "$1" >&2
    exit 1
}

# Run a command that installs something outside the repository, after asking.
# On "no", stop and print the command so it can be run by hand.
confirm_and_run() {
    local answer
    read -r -p "Run '$*' now? [y/N] " answer || answer="" # no terminal: "no"
    [[ $answer =~ ^[Yy]$ ]] || die "skipped. Run '$*' yourself, then re-run this script."
    "$@"
}

say "Checking this is an Apple-Silicon Mac"
if [[ $(uname -s) != Darwin || $(uname -m) != arm64 ]]; then
    die "needs macOS on Apple Silicon (arm64); found $(uname -sm).
       On an Apple-Silicon Mac this means the terminal runs under Rosetta."
fi

say "Checking no conda environment is active"
# An active conda environment leaks into the build: CMakeLists.txt puts
# $CONDA_PREFIX/include ahead of every other header path, and CMake finds
# packages through conda's bin/ on PATH. Anaconda's Boost, for one, then
# shadows the build's own and fails to compile against current Xcode, whose
# C++17 standard library no longer has std::unary_function.
if [[ -n ${CONDA_PREFIX:-} ]]; then
    die "conda environment $CONDA_PREFIX is active, and would leak into the build.
       Run 'conda deactivate' until no environment is active, then re-run."
fi

say "Checking the Xcode Command Line Tools (compilers, git)"
xcode-select -p >/dev/null 2>&1 ||
    die "not installed. Run 'xcode-select --install', finish the installer, then re-run."

say "Checking Homebrew"
if ! command -v brew >/dev/null && [[ -x /opt/homebrew/bin/brew ]]; then
    eval "$(/opt/homebrew/bin/brew shellenv)" # installed but not on PATH yet
fi
command -v brew >/dev/null || die "not installed. Install it from https://brew.sh, then re-run."

say "Checking Homebrew packages: ${BREW_PACKAGES[*]}"
missing=()
for pkg in "${BREW_PACKAGES[@]}"; do
    brew list --versions "$pkg" >/dev/null || missing+=("$pkg")
done
if ((${#missing[@]})); then
    confirm_and_run brew install "${missing[@]}"
fi

say "Checking uv"
command -v uv >/dev/null || confirm_and_run brew install uv

say "Fetching the bundled C++ libraries (git submodules)"
git rev-parse --git-dir >/dev/null 2>&1 ||
    die "not a git clone. Clone the repository with git rather than downloading a zip."
git submodule update --init --recursive

say "Creating .venv (Python $PYTHON_VERSION)"
if [[ -x .venv/bin/python ]]; then
    # A venv made from conda's Python runs on conda's interpreter and the
    # libraries it bundles, which this build is not tested against. Conda
    # installs are marked by a conda-meta directory.
    base="$(.venv/bin/python -c 'import sys; print(sys.base_prefix)')"
    if [[ -d $base/conda-meta ]]; then
        die ".venv was made from conda's Python ($base).
       Remove it with 'rm -rf .venv', then re-run to make it from uv's own Python."
    fi
    echo "exists, reusing it"
else
    # --managed-python: uv's own Python, never a conda or Homebrew one on PATH
    uv venv --managed-python --python "$PYTHON_VERSION" .venv
fi

say "Building and installing FIRM3D (compiles the C++/Metal extension: a few minutes)"
# --reinstall-package forces the rebuild: uv keys its build cache on the
# packaging files only, so changed C++ or Metal sources would otherwise be
# silently skipped.
#
# OpenMP stays off, even where Homebrew's libomp is installed: booz_xform's
# wheel bundles its own libomp, and with two OpenMP runtimes in one process
# macOS keeps one copy of the symbols both export, so firm3dpp's runtime runs
# code of booz_xform's, which was never started, and segfaults in its first
# multithreaded loop. OpenMP only threads the field setup's Fourier
# transforms; CPU tracing is parallel over MPI ranks.
CMAKE_ARGS="${CMAKE_ARGS:+$CMAKE_ARGS }-DCMAKE_DISABLE_FIND_PACKAGE_OpenMP=ON" \
    uv pip install --python .venv/bin/python --reinstall-package firm3d -e ".[dev]"

say "Verifying the Metal backend"
.venv/bin/python -c '
import sys, firm3dpp
backend = getattr(firm3dpp, "gpu_backend", None)
sys.exit(None if backend == "metal" else f"built without Metal (gpu_backend = {backend!r})")
'
extension="$(.venv/bin/python -c 'import firm3dpp; print(firm3dpp.__file__)')"
if otool -L "$extension" | grep -q 'libomp\.dylib'; then
    die "the extension links libomp although the build turns OpenMP off (see above)."
fi
.venv/bin/python -m pytest tests/field/test_metal.py -q

say "Verifying MPI (two ranks)"
mpiexec -n 2 .venv/bin/python -m mpi4py -c '
from mpi4py import MPI
assert MPI.COMM_WORLD.size == 2, "mpiexec did not start two ranks"
'

cat <<EOF

FIRM3D with the Metal backend is installed in $repo_root/.venv.
To benchmark this Mac, see benchmarks/README.md, or directly:

    cd benchmarks
    ../.venv/bin/python run_benchmark.py alpha-loss-wistell-a.yaml --tag <your-github-handle>
EOF
