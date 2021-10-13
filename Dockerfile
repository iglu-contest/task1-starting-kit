FROM zimli/iglu-task1:v1
FROM python:3.7
USER root
#RUN apt-get update \
#     && apt-get install -y \
#        libgl1-mesa-glx \
#        libx11-xcb1 \
#     && apt-get clean all \
     #&& rm -r /var/lib/apt/lists/*

ARG CONDA_PYTHON_VERSION=3
ARG CONDA_DIR=/opt/conda

# Install miniconda
ENV PATH $CONDA_DIR/bin:$PATH
RUN wget --quiet https://repo.continuum.io/miniconda/Miniconda3-py37_4.8.2-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    echo 'export PATH=$CONDA_DIR/bin:$PATH' > /etc/profile.d/conda.sh && \
    /bin/bash /tmp/miniconda.sh -b -p $CONDA_DIR && \
    rm -rf /tmp/*

RUN conda install --yes \
    astropy \
    matplotlib \
    pandas \
    scikit-learn \
    scikit-image

RUN conda install pytorch==1.0.1 torchvision==0.2.2 cudatoolkit=10.0 -c pytorch
USER root

ADD requirements.txt /requirements.txt

RUN pip install -r /requirements.txt

RUN python -m nltk.downloader punkt
CMD ["bash"]
