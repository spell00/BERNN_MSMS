FROM rocker/tidyverse:4.2.0

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    libcurl4-openssl-dev \
    libssl-dev \
    libxml2-dev \
    libhdf5-dev \
    libfontconfig1-dev \
    libharfbuzz-dev \
    libfribidi-dev \
    libfreetype6-dev \
    libpng-dev \
    libtiff5-dev \
    libjpeg-dev \
    libgsl-dev \
    && rm -rf /var/lib/apt/lists/*

# Install R packages
RUN R -e "install.packages(c('harmony', 'sva', 'factor', 'devtools'), repos='https://cloud.r-project.org/')" && \
    R -e "BiocManager::install('zinbwave')" && \
    R -e "devtools::install_github('immunogenomics/lisi')" && \
    R -e "install.packages('https://cran.r-project.org/src/contrib/Archive/gPCA/gPCA_1.0.tar.gz', repos=NULL, type='source')" && \
    R -e "devtools::install_github('dengkuistat/WaveICA', host='https://api.github.com')"

# Install Python packages
RUN pip3 install --no-cache-dir \
    numpy==1.23.5 \
    scikit-learn==1.0.2 \
    pandas==1.4.4 \
    scikit-optimize==0.9.0 \
    matplotlib==3.6.3 \
    seaborn==0.12.2 \
    tabulate==0.9.0 \
    scipy==1.9.1 \
    tqdm \
    joblib~=1.0.0 \
    ax-platform==0.2.10 \
    pycombat \
    torch>=2.3.0 \
    torchvision>=0.15.0 \
    tensorboardX>=2.5.1 \
    tensorboard==2.11.0 \
    tensorflow==2.11.0 \
    psutil==5.9.4 \
    scikit-image \
    nibabel \
    mpmath==1.3.0 \
    patsy==0.5.3 \
    umap-learn==0.5.3 \
    shapely==2.0.0 \
    numba==0.57.1 \
    rpy2==3.5.7 \
    openpyxl==3.0.10 \
    xgboost==1.0.0 \
    torch-geometric \
    neptune \
    fastapi==0.89.1 \
    "mlflow[extras]" \
    threadpoolctl==3.1.0 \
    protobuf==3.20.1 \
    shap

# Create app directory
WORKDIR /app

# Set environment variables
ENV PATH="/usr/local/bin:$PATH"
ENV PYTHONPATH="/app:$PYTHONPATH"
ENV R_LIBS="/usr/local/lib/R/site-library"

# Default command
CMD ["/bin/bash"]