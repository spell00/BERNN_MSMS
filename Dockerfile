FROM ubuntu:20.04

ADD mlflow_eval_runs.py ./
ADD launch_train_ae_classifier_holdout_experiments.sh ./
ADD launch_train_ae_then_classifier_holdout_experiments.sh ./
ADD mlflow_eval_runs.py ./
ADD src ./src/
COPY requirements.txt ./requirements.txt
ADD data ./data/
ADD notebooks ./notebooks/
RUN chmod +x launch_train_ae_classifier_holdout_experiments.sh
RUN chmod +x launch_train_ae_then_classifier_holdout_experiments.sh

RUN apt-get update
RUN apt-get upgrade -y

ENV TZ=America/Toronto
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt-get install -y r-base r-cran-devtools python3-pip python3-dev \
  && cd /usr/local/bin \
  && ln -s /usr/bin/python3 python \
  && pip3 install --upgrade pip

RUN R -e "devtools::install_github('immunogenomics/lisi', host='https://api.github.com')"
# RUN R -e 'devtools::install_github("dengkuistat/WaveICA", host="https://api.github.com")'
# RUN R -e "devtools::install_github('zinbwave')"
# RUN R -e "BiocManager::install_github('zinbwave')"
RUN R -e "install.packages('harmony',dependencies=TRUE, repos='http://cran.rstudio.com/')"
RUN R -e "install.packages('sva',dependencies=TRUE, repos='http://cran.rstudio.com/')"
RUN R -e "install.packages('factor',dependencies=TRUE, repos='http://cran.rstudio.com/')"
# RUN R -e "install.packages('devtools')"
RUN R -e "install.packages('https://cran.r-project.org/src/contrib/Archive/gPCA/gPCA_1.0.tar.gz', repos = NULL, type = 'source')"

RUN python -m pip install -r requirements.txt

# CMD ./mzdb2train.sh test


# R packages:
# BiocManager::install("zinbwave")
# devtools::install_github("immunogenomics/lisi")
# devtools::install_github("dengkuistat/WaveICA",host="https://api.github.com")
