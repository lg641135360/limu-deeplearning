1. 首先进入宿主机上的工作目录

   1. ```bash
      cd /mnt/sda1/rikoo
      ```

2. 基于官方的anaconda3镜像实例化一个本地容器

   1. ```bash
      docker run -it --gpus all --name rikoo_py38 --ipc=host -p 9711:9711 -v `pwd`:/root -e NVIDIA_DRIVER_CAPABILITIES=compute,utility -e NVIDIA_VISIBLE_DEVICES=all continuumio/anaconda3 /bin/bash
      ```

      1. --name：容器名称
      2. -p 8888:8888：将容器的8888端口映射到本地的8888端口，便于访问jupyter
      3. -it：用交互的方式打开容器
      4. -v 将宿主机当前目录pwd挂载到容器内/root目录，容器内对/root进行操作会同步到宿主机
      5. --ipc=host 与宿主机共享内存
      6. 其他参数是为了在容器内使用英伟达显卡驱动

3. 更换源

   1. docker容器中没有安装vim等编辑工具，安装时查找不了包，需要更换国内源	

   2. ```bash
      apt-get update		
      apt-get install vim
      ```

   3. 更换linux默认软件源

      1. ```bash
         vim /etc/apt/sources.list
         
         
         
         # 更新源
         apt-get update 
         apt-get upgrade
         ```
   
4. 更改conda源

   1. ```bash
      vim ~/.condarc
      
      channels:
        - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/
        - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
        - https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge/
        - https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch/
      ssl_verify: true
      ```

5. 安装jupyterLab

   1. ```bash
      conda install -c conda-forge jupyterlab
      ```

6. 启动jupyter lab

   1. ```bash
      cd ~
      jupyter lab --ip='*' --port=9711 --no-browser --allow-root
      ```

7. 退出容器

   1. ```bash
      ctrl+c退出笔记本
      exit
      ```

8. 重启动容器

   1. ```bash
      # 启动容器
      docker start rikoo_py38
      docker exec -it rikoo_py38 /bin/bash
      ```



#### 结束

