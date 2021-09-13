#### 本地安装（cpu版本）

* [可选] 使用conda/miniconda环境
  * conda env remove d2l-zh
  * conda create -n d2l-zh python=3.8
  * conda activate d2l-zh
* 安装需要的包
  * pip install torch===1.5.1 torchvision===0.6.1 -f https://download.pytorch.org/whl/torch_stable.html -i https://pypi.douban.com/simple
  * pip install d2l
  * 使用jupyter lab替代jupyter notebook
  * conda install -c conda-forge jupyterlab
* 下载代码并执行
  * wget https://zh-v2.d2l.ai/d2l-zh.zip
  * unzip d2l-zh.zip
  * jupyter notebook

