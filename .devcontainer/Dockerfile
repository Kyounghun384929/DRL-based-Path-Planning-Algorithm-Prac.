FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive TZ=Asia/Seoul

RUN apt-get update -y && \
    apt-get install -y --no-install-recommends \
        build-essential \
        apt-utils \
        ca-certificates \
        tzdata \
        libsm6 libxrender1 libxext6 libglib2.0-0 \
        sudo git curl wget vim tree htop unzip \
        libssl-dev libffi-dev libbz2-dev libreadline-dev libsqlite3-dev \
        libncurses5-dev libncursesw5-dev xz-utils tk-dev \
        libxml2-dev libxmlsec1-dev liblzma-dev zlib1g-dev \
        libgdbm-dev libnss3-dev libgdbm-compat-dev \
        libgmp-dev libmpfr-dev libmpc-dev \
        dvipng \
        texlive-latex-base texlive-latex-recommended \
        texlive-fonts-recommended texlive-latex-extra \
        cm-super \
        fonts-noto-cjk \
        fontconfig && \
    ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone && \
    rm -rf /var/lib/apt/lists/*

RUN fc-cache -fv

ARG PYTHON_VERSION=3.13.9
ARG PYTHON_MM_VERSION=3.13

# Python 소스 컴파일 설치
RUN cd /tmp && \
    wget https://www.python.org/ftp/python/${PYTHON_VERSION}/Python-${PYTHON_VERSION}.tgz && \
    tar -xzf Python-${PYTHON_VERSION}.tgz && \
    cd Python-${PYTHON_VERSION} && \
    ./configure \
        --enable-optimizations \
        --enable-shared \
        --with-ensurepip=install \
        --prefix=/usr/local/python-${PYTHON_VERSION} && \
    make -j$(nproc) && \
    make install && \
    cd / && \
    rm -rf /tmp/Python-${PYTHON_VERSION}*
# Python 연결
RUN update-alternatives --install /usr/bin/python3 python3 /usr/local/python-${PYTHON_VERSION}/bin/python${PYTHON_MM_VERSION} 10 && \
    update-alternatives --config python3 || true && \
    update-alternatives --install /usr/bin/python python /usr/local/python-${PYTHON_VERSION}/bin/python${PYTHON_MM_VERSION} 10 && \
    update-alternatives --config python || true && \
    update-alternatives --install /usr/bin/pip3 pip3 /usr/local/python-${PYTHON_VERSION}/bin/pip${PYTHON_MM_VERSION} 10 && \
    update-alternatives --config pip3 || true && \
    update-alternatives --install /usr/bin/pip pip /usr/local/python-${PYTHON_VERSION}/bin/pip${PYTHON_MM_VERSION} 10 && \
    update-alternatives --config pip || true

# Python 라이브러리 경로 설정
RUN echo "/usr/local/python-${PYTHON_VERSION}/lib" > /etc/ld.so.conf.d/python${PYTHON_MM_VERSION}.conf && ldconfig
# Python 공유 라이브러리 경로 설정
ENV LD_LIBRARY_PATH="/usr/local/python-${PYTHON_VERSION}/lib"

# pip 업그레이드
RUN pip install --upgrade --no-cache-dir pip setuptools wheel && \
    rm -rf /root/.cache/pip

# Python 3.13 빌드 이후
RUN apt-get purge -y \
    libssl-dev libffi-dev libbz2-dev libreadline-dev libsqlite3-dev \
    libncurses5-dev libncursesw5-dev xz-utils tk-dev \
    libxml2-dev libxmlsec1-dev liblzma-dev zlib1g-dev \
    libgdbm-dev libnss3-dev libgdbm-compat-dev \
    libgmp-dev libmpfr-dev libmpc-dev && \
    apt-get autoremove -y && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Pytorch 설치
RUN pip install --upgrade --no-cache-dir pip setuptools wheel
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
RUN rm -rf /root/.cache/pip

# 사용자 추가
ARG USERNAME=vscode
ARG USER_UID=1001
ARG USER_GID=$USER_UID

ENV USERNAME=${USERNAME}
ENV USER_UID=${USER_UID}
ENV USER_GID=${USER_GID}

RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    && echo "$USERNAME ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME

# 작업 디렉토리 권한 설정
RUN mkdir -p /workspaces && \
    chown -R $USERNAME:$USERNAME /workspaces && \
    chmod -R u+w /workspaces

# PATH 설정 
ENV PATH="/home/${USERNAME}/.local/bin:${PATH}"

WORKDIR /workspaces

# 최종 사용자 전환
USER $USERNAME

# 로그인 쉘로 시작
CMD [ "bash", "-l" ]