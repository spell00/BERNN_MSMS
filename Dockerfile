FROM nvidia/cuda:12.1.1-runtime-ubuntu20.04 

# Set environment variables to prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=America/Toronto
# Combine all apt-get commands into a single RUN to reduce layers
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3.8 \
        python3.8-dev \
        python3.8-distutils \
        curl \
        libssl-dev \
        libcurl4-openssl-dev \
        libfontconfig1-dev \
        r-base \
        r-base-dev \
        r-cran-devtools \
        python3-pip \
        python3-dev \
        software-properties-common \
        linux-modules-nvidia-525-generic && \
    ln -sf /usr/bin/python3.8 /usr/bin/python && \
    curl -sS https://bootstrap.pypa.io/pip/3.8/get-pip.py | python && \
    pip install --upgrade pip && \
    ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && \
    echo $TZ > /etc/timezone && \
    rm -rf /var/lib/apt/lists/*

# Add your files
ADD mlflow_eval_runs.py ./
ADD setup.py ./
ADD launch_train_ae_classifier_holdout_experiments.sh ./
ADD launch_train_ae_then_classifier_holdout_experiments.sh ./
ADD mlflow_eval_runs.py ./
ADD bernn ./bernn/
ADD tests ./tests/
COPY requirements.txt ./requirements.txt
COPY README.md ./README.md
# ADD data ./data/
# ADD notebooks ./notebooks/

# Make scripts executable
RUN chmod +x launch_train_ae_classifier_holdout_experiments.sh && \
    chmod +x launch_train_ae_then_classifier_holdout_experiments.sh

# Install R packages
RUN R -e "install.packages('ragg')" 
RUN R -e "install.packages('pkgdown')" 
RUN R -e "install.packages('devtools')"
# RUN R -e "devtools::install_github('immunogenomics/lisi', host='https://api.github.com')"
# RUN R -e 'devtools::install_github("dengkuistat/WaveICA", host="https://api.github.com")'
# RUN R -e "devtools::install_github('zinbwave')"
# RUN R -e "BiocManager::install_github('zinbwave')"
# RUN R -e "install.packages('harmony',dependencies=TRUE, repos='http://cran.rstudio.com/')"
RUN R -e "install.packages('BiocManager')" 

# Install Python packages
RUN python -m pip install -r requirements.txt && \
    python setup.py build && \
    python setup.py install && \
    python -m pip install .

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
