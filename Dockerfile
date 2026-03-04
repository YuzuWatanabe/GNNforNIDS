# 基本イメージとして continuumio/anaconda3 を使用
FROM continuumio/anaconda3:latest

# 必要なツールのインストール
RUN apt-get update && apt-get install -y \
    wget \
    gnupg2 \
    git \
    build-essential \
    cmake \
    wireshark \
    tshark \
    libpcap0.8-dev \
    libgnutls30 \
    libwireshark-dev \
    libssl-dev \
    && apt-get clean \dgl\
    && rm -rf /var/lib/apt/lists/*

# Monoのリポジトリを追加して、monoをインストール
RUN apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv-keys 3FA7E0328081BFF6A14DA29AA6A19B38D3D831EF \
    && echo "deb https://download.mono-project.com/repo/ubuntu stable-focal main" | tee /etc/apt/sources.list.d/mono-official-stable.list \
    && apt-get update && apt-get install -y mono-complete \
    && rm -rf /var/lib/apt/lists/*

# ローカルのSplitCap.exeをコンテナにコピー
COPY SplitCap.exe /workspace/SplitCap.exe

# Conda 環境の作成
RUN conda create -n anaconda3-cuda python=3.8 && \
    conda clean -a -y

# Conda 環境でパッケージのインストール
RUN /bin/bash -c "source /opt/conda/etc/profile.d/conda.sh && \
    conda activate anaconda3-cuda && \
    conda config --add channels defaults && \
    conda config --add channels conda-forge && \
    conda config --add channels pytorch && \
    conda config --add channels dglteam && \
    conda install -y cudatoolkit=11.8 && \
    conda install pytorch=2.3 torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia && \
    conda install -y scipy numpy && \
    conda install pandas &&\
    conda install matplotlib &&\
    conda install torchdata &&\
    conda install scikit-learn &&\
    conda install pydantic &&\
    conda install dglteam/label/th23_cu118::dgl &&\
    conda clean -a -y &&\
    conda install scapy &&\
    pip install pyshark"

# SplitCapのヘルプを表示して動作確認
RUN mono /workspace/SplitCap.exe

# 環境変数の設定
ENV PATH=/opt/conda/envs/anaconda3-cuda/bin:$PATH
ENV LD_LIBRARY_PATH=/opt/conda/envs/anaconda3-cuda/lib:/usr/local/cuda/lib64:$LD_LIBRARY_PATH
ENV CUDA_HOME=/usr/local/cuda
ENV DGLBACKEND=pytorch
#ENV TSHARK_PATH=/usr/bin:$PATH 

# JupyterLab のインストール
RUN /bin/bash -c "source /opt/conda/etc/profile.d/conda.sh && \
    conda activate anaconda3-cuda && \
    conda install -c conda-forge jupyterlab ipywidgets && \
    pip install jupyterlab_widgets && \
    conda clean -a -y"

# ポート 8888 を開放
EXPOSE 8888

# コンテナ起動時に仮想環境をアクティベートし、JupyterLab を実行
CMD ["bash", "-c", "nvidia-smi && source /opt/conda/etc/profile.d/conda.sh && conda activate anaconda3-cuda && jupyter lab --ip=0.0.0.0 --allow-root --NotebookApp.token=''"]
