# AnimeGANv2

## datasets
训练数据结构设计  
|_datasets  
|__animegan  
|___Shinkai(画家风格)  
|_____train(训练数据)  
|________real(真实数据)  
|________style(风格数据)  
|________smooth(经过处理的风格数据)  
|_____test(测试数据)  
|________real(真实数据)  

## graph
```
# 视化
python scripts/modelTensorboard.py
tensorboard --logdir=./graph
```

## train
```
# 单卡
python tools/train_net.py \
--config-file "path/to/config" \
SOLVER.IMS_PER_BATCH 8

# 多卡
python -m torch.distributed.launch --nproc_r_node=8 \
/tools/train_net.py \
--config-file "path/to/config" \
SOLVER.IMS_PER_BATCH 8
```