FROM nvcr.io/nvidia/tritonserver:23.01-py3

LABEL author="zhoudailin"

WORKDIR /root

RUN pip3 install jupyterlab

COPY jupyter_lab_config.py ~/.jupyter/jupyter_lab_config.py

EXPOSE 8888

ENTRYPOINT ["jupyter", "lab", "--ip=0.0.0.0", "--allow-root", "--no-browser"]