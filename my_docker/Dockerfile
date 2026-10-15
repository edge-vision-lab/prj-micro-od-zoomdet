ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG USERNAME
ARG GROUPNAME
ARG UID
ARG GID
ARG PASSWORD

RUN apt update -y && \
    apt install -y emacs emacs-mozc emacs-mozc-bin && \
    apt install -y libcanberra-gtk-module libcanberra-gtk3-module && \
    apt install -y sudo && \
    apt install -y python3-venv && \
    apt install -y fonts-noto-cjk && \
    apt install -y libgl1-mesa-glx libglib2.0-0

RUN userdel -r ubuntu 2>/dev/null || true && groupdel ubuntu 2>/dev/null || true
RUN groupadd -g $GID $GROUPNAME && \
    useradd -m -s /bin/bash -u $UID -g $GID -G sudo $USERNAME && \
    echo $USERNAME:$PASSWORD | chpasswd && \
    echo "$USERNAME   ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

USER $USERNAME

# --- MMDetection環境構築 ---
ENV VENV_PATH="/home/${USERNAME}/venv"
ENV PATH="${VENV_PATH}/bin:$PATH"

# 仮想環境の作成（NVIDIAコンテナのPyTorch環境を引き継ぐ）
RUN python -m venv ${VENV_PATH} --system-site-packages

# MMCVなどを強制的にCUDAでコンパイルさせる
ENV FORCE_CUDA="1"

# 元のイメージの環境を壊さないように必要なものをインストール（そのためnumpyを固定）
# その他インストール可能にするためのいくつかの工夫
#   setuptoolsのダウングレード
#   mmcvの隔離ビルド無効化（--no-build-isolation）
#   通常の opencv-python をインストール（変わりに↑libgl1-mesa-glx libglib2.0-0をインストール）
RUN NUMPY_VER=$(python -c "import numpy; print(numpy.__version__)") && \
    pip install --upgrade pip "setuptools<70" && \
    pip install openmim "numpy==$NUMPY_VER" && \
    mim install mmengine "numpy==$NUMPY_VER" && \
    pip install "mmcv==2.1.0" "numpy==$NUMPY_VER" --no-build-isolation && \
    pip install opencv-python "numpy==$NUMPY_VER"

# mmdetection本体はここではインストールせず，githubから落としてきて，
# コンテナ起動時にインストールする

RUN echo 'alias lsf="ls -F"' >> ~/.bashrc && \
    echo 'alias lsa="ls -a"' >> ~/.bashrc

WORKDIR /home/$USERNAME

CMD ["/usr/bin/bash"]
