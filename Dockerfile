FROM ghcr.io/jorgensd/dolfinx_mpc:v0.10.1

WORKDIR /workspace
COPY . /workspace

RUN python3 -m pip install --no-cache-dir matplotlib pyvista pytest

CMD ["python3", "Optimus.py", "--help"]
