# medmae
# 运行代码示例

以下是一个示例命令，用于在后台运行Python脚本并将输出重定向到日志文件中：

```bash
nohup python main.py configs/mae3d_btcv_1gpu.yaml --mask_ratio=0.50 --run_name='mae3d_sincos_vit_base' > log.txt 2>&1 &
