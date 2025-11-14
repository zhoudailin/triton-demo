FROM nvcr.io/nvidia/tritonserver:23.01-py3

LABEL author="zhoudailin"

WORKDIR /workspace

COPY wheel wheel
COPY triton model

RUN pip3 install funasr jupyterlab torchvision torchaudio wheel/kaldifeat-1.25.5.dev20240914+cuda12.1.torch2.4.1-cp38-cp38-manylinux_2_17_x86_64.manylinux2014_x86_64.whl

COPY jupyter_lab_config.py ~/.jupyter/jupyter_lab_config.py

EXPOSE 8888

ENTRYPOINT ["jupyter", "lab", "--ip=0.0.0.0", "--allow-root", "--no-browser"]