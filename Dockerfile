FROM nvcr.io/nvidia/tritonserver:23.10-py3

LABEL author="zhoudailin"

WORKDIR /workspace

COPY triton model

RUN pip3 install jupyterlab kaldi-native-fbank

COPY jupyter_lab_config.py ~/.jupyter/jupyter_lab_config.py

EXPOSE 8888

ENTRYPOINT ["jupyter", "lab", "--ip=0.0.0.0", "--allow-root", "--no-browser"]