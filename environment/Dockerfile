# hash:sha256:3c1e52ab1a2bfc296cdf6b39fae2f0d65735e7192132e7914c394eb8ae8b181a
FROM registry.codeocean.com/codeocean/miniconda3:4.12.0-python3.9-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive

RUN conda install -y \
        cartopy==0.22.0 \
        cftime==1.6.4 \
        geopandas==0.14.2 \
        h5netcdf==1.2.0 \
        netcdf4==1.6.2 \
        numpy==1.26.0 \
        pandas==2.1.1 \
        pyproj==3.6.1 \
        pytz==2022.1 \
        scikit-learn==1.5.2 \
        scipy==1.11.4 \
        shapely==2.0.6 \
        xarray==2023.6.0 \
        zarr==2.13.3 \
    && conda clean -ya
