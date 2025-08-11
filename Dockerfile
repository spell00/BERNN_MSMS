FROM nvidia/cuda:12.1.1-runtime-ubuntu20.04

# Build argument for Python version
ARG PYTHON_VERSION=3.10

# Set environment variables to prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=America/Toronto

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        curl \
        libssl-dev \
        libcurl4-openssl-dev \
        libfontconfig1-dev \
        libfreetype6-dev \
        libpng-dev \
        libtiff5-dev \
        libjpeg-dev \
        libharfbuzz-dev \
        libfribidi-dev \
        libxml2-dev \
        libgit2-dev \
        r-base \
        r-base-dev \
        r-cran-devtools \
        python3-dev \
        software-properties-common \
        linux-modules-nvidia-525-generic && \
    ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && \
    echo $TZ > /etc/timezone && \
    rm -rf /var/lib/apt/lists/*

# Install Python version and upgrade pip
RUN if [ "$PYTHON_VERSION" = "3.8" ]; then \
        apt-get update && \
        apt-get install -y --no-install-recommends python3.8 python3.8-dev python3.8-distutils && \
        ln -sf /usr/bin/python3.8 /usr/bin/python && \
        ln -sf /usr/bin/python3.8 /usr/bin/python3 && \
        curl -sS https://bootstrap.pypa.io/get-pip.py -o get-pip.py && \
        python3.8 get-pip.py && \
        python3.8 -m pip install --upgrade pip setuptools wheel && \
        rm get-pip.py; \
    elif [ "$PYTHON_VERSION" = "3.10" ]; then \
        add-apt-repository ppa:deadsnakes/ppa && \
        apt-get update && \
        apt-get install -y --no-install-recommends python3.10 python3.10-dev python3.10-distutils && \
        ln -sf /usr/bin/python3.10 /usr/bin/python && \
        ln -sf /usr/bin/python3.10 /usr/bin/python3 && \
        curl -sS https://bootstrap.pypa.io/get-pip.py -o get-pip.py && \
        python3.10 get-pip.py && \
        python3.10 -m pip install --upgrade pip setuptools wheel && \
        rm get-pip.py; \
    elif [ "$PYTHON_VERSION" = "3.11" ]; then \
        add-apt-repository ppa:deadsnakes/ppa && \
        apt-get update && \
        apt-get install -y --no-install-recommends python3.11 python3.11-dev python3.11-distutils && \
        ln -sf /usr/bin/python3.11 /usr/bin/python && \
        ln -sf /usr/bin/python3.11 /usr/bin/python3 && \
        curl -sS https://bootstrap.pypa.io/get-pip.py -o get-pip.py && \
        python3.11 get-pip.py && \
        python3.11 -m pip install --upgrade pip setuptools wheel && \
        rm get-pip.py; \
    elif [ "$PYTHON_VERSION" = "3.12" ]; then \
        add-apt-repository ppa:deadsnakes/ppa && \
        apt-get update && \
        apt-get install -y --no-install-recommends python3.12 python3.12-dev && \
        ln -sf /usr/bin/python3.12 /usr/bin/python && \
        ln -sf /usr/bin/python3.12 /usr/bin/python3 && \
        curl -sS https://bootstrap.pypa.io/get-pip.py -o get-pip.py && \
        python3.12 get-pip.py && \
        python3.12 -m pip install --upgrade pip setuptools wheel && \
        rm get-pip.py; \
    else \
        apt-get update && \
        apt-get install -y --no-install-recommends python3 python3-dev python3-distutils && \
        python3 -m pip install --upgrade pip setuptools wheel; \
    fi

# Add your files
ADD mlflow_eval_runs.py ./
ADD setup.py ./
ADD launch_train_ae_classifier_holdout_experiments.sh ./
ADD launch_train_ae_then_classifier_holdout_experiments.sh ./
ADD mlflow_eval_runs.py ./
ADD bernn ./bernn/
ADD tests ./tests/
ADD test_python_versions.py ./
# ADD resolve_python38_conflicts.py ./
# ADD resolve_conflicts.py ./
COPY requirements.txt ./requirements.txt
COPY README.md ./README.md
COPY INSTALLATION.md ./INSTALLATION.md
# ADD data ./data/
# ADD notebooks ./notebooks/

# Make scripts executable
RUN chmod +x launch_train_ae_classifier_holdout_experiments.sh && \
    chmod +x launch_train_ae_then_classifier_holdout_experiments.sh

# Install R packages with proper dependency handling
RUN R -e "install.packages(c('cpp11', 'systemfonts', 'textshaping', 'ragg', 'pkgdown', 'devtools', 'BiocManager'), dependencies=TRUE, repos='https://cloud.r-project.org/')"

# Install Python packages with version-specific logic
RUN echo "Installing for Python $PYTHON_VERSION" && \
    # ...existing code...
    if [ "$PYTHON_VERSION" = "3.8" ]; then \
        echo "Installing Python 3.8 specific packages..." && \
        pip install .[python38-full]; \
    elif [ "$PYTHON_VERSION" = "3.10" ]; then \
        echo "Installing Python 3.10 specific packages..." && \
        pip install .[full]; \
    elif [ "$PYTHON_VERSION" = "3.11" ]; then \
        echo "Installing Python 3.11 specific packages..." && \
        pip install .[python311-plus]; \
    elif [ "$PYTHON_VERSION" = "3.12" ]; then \
        echo "Installing Python 3.12 specific packages..." && \
        pip install .[py312-plus]; \
    else \
        echo "Installing default packages..." && \
        pip install .; \
    fi

ENV R_HOME=/usr/lib/R
ENV LD_LIBRARY_PATH=/usr/lib/R/lib:${LD_LIBRARY_PATH}
# CMD ./mzdb2train.sh test
# CMD ["pytest", "-v", "-rs", "--cov=bernn", "--cov-report=term", "--cov-report=xml:coverage.xml", "tests/"]




# RUN R -e "BiocManager::install('sva')"
# RUN R -e "install.packages('sva',dependencies=TRUE, repos='http://cran.rstudio.com/')"
# BiocManager::install(c("GenomeInfoDb", "Biostrings", "KEGGREST", "AnnotationDbi", "annotate", "genefilter"))
# BiocManager::install("sva")
# RUN R -e "install.packages('factor',dependencies=TRUE, repos='http://cran.rstudio.com/')"
# RUN R -e "install.packages('https://cran.r-project.org/src/contrib/Archive/gPCA/gPCA_1.0.tar.gz', repos = NULL, type = 'source')"
