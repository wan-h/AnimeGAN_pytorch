# AnimeGANv2

## Docker
```
# 构建镜像
docker build -t animeganv2:dev . --network host

# 启动镜像
nvidia-docker run \
--rm \
--name=animeganv2 \
-v `pwd`:/workspace \
-v /data:/data \
--privileged=true \
-it animeganv2:dev bash

# 安装依赖库
pip install --no-cache -r requirements.txt
```