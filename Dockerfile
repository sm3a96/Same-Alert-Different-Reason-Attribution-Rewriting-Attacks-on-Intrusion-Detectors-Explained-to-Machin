# Reproduction container for EIB.
#
# Scope, stated up front so nobody is surprised. This image reproduces everything that runs on
# CPU: the attack suite, the integrity signals, the conformal calibration, the C3 matrix, and
# every figure and table. It does NOT run the C2 judges -- those need a GPU and two open-weights
# models, and baking 20+ GB of weights into an image is the wrong shape for an artifact. The
# cached judge decisions ship with the release, so `make floats` regenerates every C2 float from
# them without a GPU.
#
# Datasets are not redistributed and are not in the image. Mount them, or run
# `python scripts/download_data.py` inside the container after accepting each provider's terms.
#
#   docker build -t xintbench .
#   docker run --rm -v "$PWD/data:/work/data" -v "$PWD/results:/work/results" xintbench make paper
#
# torch is pinned to 2.4.0 by constraints.txt and must stay there: `rtdl` and `tabpfn` drag it
# back to 1.13.1, and transformers v5 imports a symbol 2.4.0 does not have.

FROM python:3.10-slim

# tectonic builds Fig1 (TikZ). It pulls its own TeX packages on first run, so the image stays
# small; a build with no network simply skips Fig1 and `make fig1` says so rather than failing.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential git curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /work

# Dependencies first, so a source edit does not invalidate the pip layer.
COPY pyproject.toml constraints.txt requirements.lock.txt ./
RUN pip install --no-cache-dir -r requirements.lock.txt -c constraints.txt

COPY . .
RUN pip install --no-cache-dir -e . -c constraints.txt

# Fail early and loudly if the pin slipped during the build.
RUN python -c "import torch, sys; v = torch.__version__.split('+')[0]; \
    sys.exit(0) if v == '2.4.0' else sys.exit(f'torch {v}, expected 2.4.0 -- a dependency \
re-resolved it and the FT-Transformer and judge paths will not behave')"

RUN python -m pytest -q

CMD ["make", "paper"]
