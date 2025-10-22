##### 镜像构建
docker build . -t triton-jupyter

##### 镜像执行
docker run --name triton-jupyter -p 8888:8888 -v $(pwd)/triton:/workspace/model -d triton-jupyter