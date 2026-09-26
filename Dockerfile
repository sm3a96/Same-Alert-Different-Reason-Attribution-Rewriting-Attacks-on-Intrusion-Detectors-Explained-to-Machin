# Reproduction container for EIB.
#
# Scope, stated up front so nobody is surprised. This image reproduces everything that runs on
# CPU: the attack suite, the integrity signals, the conformal calibration, the detectability matrix, and
# every figure and table. It does NOT run the reader models -- those need a GPU and two open-weights
# models, and baking 20+ GB of weights into an image is the wrong shape for an artifact. The
# cached judge decisions ship with the release, so `make floats` regenerates every reader-harm float from
# them without a GPU.
#
# Datasets are not redistributed and are not in the image. Mount them, or run
# `python scripts/download_data.py` inside the container after accepting each provider's terms.
#
#   docker build -t eib .
#   docker run --rm -v "$PWD/results:/work/results" eib make floats
#   docker run --rm eib make test
#
# torch is pinned to 2.4.0 by constraints.txt and must stay there: `rtdl` and `tabpfn` drag it
# back to 1.13.1, and transformers v5 imports a symbol 2.4.0 does not have.

FROM python:3.10-slim

# The shipped targets are `make setup`, `make test`, `make floats` and `make verify`; every
# figure in figures/out/ ships as a PDF, so no TeX toolchain is needed to reproduce them.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential git curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /work

# Dependencies first, so a source edit does not invalidate the pip layer.
COPY pyproject.toml constraints.txt requirements.lock.txt ./
RUN pip install --no-cache-dir -r requirements.lock.txt -c constraints.txt \
        --extra-index-url https://download.pytorch.org/whl/cu124

COPY . .
RUN pip install --no-cache-dir -e . -c constraints.txt

# Fail early and loudly if the pin slipped during the build.
RUN python -c "import torch, sys; v = torch.__version__.split('+')[0]; \
    sys.exit(0) if v == '2.4.0' else sys.exit(f'torch {v}, expected 2.4.0 -- a dependency \
re-resolved it and the FT-Transformer and judge paths will not behave')"

RUN python -m pytest -q

CMD ["make", "floats"]
