##### 镜像构建
docker build . -t triton-jupyter

##### 镜像执行

```shell
docker run --name triton-jupyter --ipc=host --shm-size=1g --gpus all -p 8888:8888 -p 8000:8000 -p 8001:8001 -p 8002:8002 -v $(pwd)/triton:/workspace/model -d  triton-jupyter
```

```powershell
docker run --name triton-jupyter --ipc=host --shm-size=1g --gpus all -p 8888:8888 -p 8000:8000 -p 8001:8001 -p 8002:8002 -v "$(Get-Location)\triton:/workspace/model" -d triton-jupyter
```

> 注意：在 Linux 上运行容器时必须添加 `--ipc=host`，否则 Python Backend 与 GPU 模型之间无法使用 CUDA IPC 共享显存，推理时会报 `Failed to open the cudaIpcHandle. error: invalid resource handle`。
