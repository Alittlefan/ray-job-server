#!/bin/bash

# 设置脚本在任何命令失败时退出
set -e

#pip install seaborn scikit-learn git+https://gitee.com/Morphlng/tcp_comm.git
# 安装 TensorBoard
#echo "[$(date)] Installing TensorBoard..."
#pip install tensorboard
cp /workspace/docker/webfiles.zip /home/ray/anaconda3/lib/python3.8/site-packages/tensorboard/
rm -rf /home/ray/anaconda3/lib/python3.8/site-packages/ray/dashboard/client/build
cp /workspace/docker/build.tar.gz /home/ray/anaconda3/lib/python3.8/site-packages/ray/dashboard/client/
cd /home/ray/anaconda3/lib/python3.8/site-packages/ray/dashboard/client/
tar -zxvf build.tar.gz

# 启动 TensorBoard
echo "[$(date)] Starting TensorBoard..."
nohup tensorboard --logdir=/nfsdata/ray/ray_results --host=0.0.0.0 --port=6006 > /dev/null 2>&1 &


# 安装依赖
#echo "[$(date)] Installing Python dependencies from requirements.txt..."
#pip install -r requirements.txt

# 启动 server.py
cd /workspace
sleep 10
echo "[$(date)] Starting server.py..."
TENSORBOARD_ADDRESS="http://192.168.16.36:6006/" python server.py > server.log 2>&1 &

echo "[$(date)] server.py started successfully."
