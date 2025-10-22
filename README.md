##### 镜像构建
docker build . -t triton-jupyter

##### 镜像执行

```shell
docker run --name triton-jupyter --gpus all -p 8888:8888 -v $(pwd)/triton:/workspace/model -d  triton-jupyter
```

```powershell
docker run --name triton-jupyter --gpus all -p 8888:8888 -v "$(Get-Location)\triton:/workspace/model" -d triton-jupyter
```