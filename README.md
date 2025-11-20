##### 镜像构建
docker build . -t triton-jupyter

##### 镜像执行

```shell
docker run --name triton-jupyter --shm-size=1g --gpus all -p 8888:8888 -p 8000:8000 -p 8001:8001 -p 8002:8002 -v $(pwd)/triton:/workspace/model -d  triton-jupyter
```

```powershell
docker run --name triton-jupyter --shm-size=1g --gpus all -p 8888:8888 -p 8000:8000 -p 8001:8001 -p 8002:8002 -v "$(Get-Location)\triton:/workspace/model" -d triton-jupyter
```