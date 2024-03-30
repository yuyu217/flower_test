**This project is based on** [ASTGCN-2019-pytorch](https://github.com/guoshnBJTU/ASTGCN-2019-pytorch) and [FlowerFL Quickstart PyTorch](https://github.com/adap/flower)

## 環境
* Python版本：3.10 (PyTorch無法支援更高版本，3.9版會Segment Fault)
* GPU： NVIDIA GPU, AMD GPU(使用 ROCm或ZLUDA),  支援Metal Performance Shaders的Mac

## client.py
### 參數
``--config``：配置檔路徑
``--partition-id``：分割編號，起始於零
``--partition-size``：分割數量
``--force-cpu``：強制使用cpu
``--cuda-device-id``：CUDA裝置編號

## server.py
### 參數
``--num_rounds``：回合數

## PEMS08_FL batch_size設定
對於運行3個客戶端，

``batch_size=48`` 消耗顯示記憶體約爲7.7GiB
``batch_size=32`` 消耗顯示記憶體約爲4.4GiB
``batch_size=24`` 消耗顯示記憶體約爲3.0GiB



 
