FROM nvcr.io/nvidia/tritonserver:23.10-py3

LABEL author="zhoudailin"

WORKDIR /workspace

RUN pip3 install jupyterlab torch==1.13.1 torchaudio==0.13.1

COPY wheel /opt/app/wheel

RUN pip3 install /opt/app/wheel/kaldifeat-1.25.5.dev20250630+cpu.torch1.13.1-cp310-cp310-manylinux2014_x86_64.manylinux_2_17_x86_64.whl

COPY triton model

COPY jupyter_lab_config.py ~/.jupyter/jupyter_lab_config.py

EXPOSE 8888

ENTRYPOINT ["jupyter", "lab", "--ip=0.0.0.0", "--allow-root", "--no-browser"]