# 已迁移：飞行器布局 VAE 定义

本文件不再定义当前飞行器训练接口。原有内容中仍有效的树节点、后序 `ops`、8D symmetry、typed 序列编码、递归 encoder/decoder 和 torchfold 约定，已迁入 [分层飞行器联合 AE 编码定义](hierarchical_aircraft_encoding.md)。

当前飞行器训练使用确定性 AE，入口为 `train_autoencoder.py`，不使用 VAE sampler 或 KL divergence。旧 13D OBB VAE/GAN 代码只保留为历史参考，不能与 typed 飞行器数据集混用。
