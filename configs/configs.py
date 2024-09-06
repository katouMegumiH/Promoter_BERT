import os
print(os.getcwd())  # 查看当前工作目录
max_length = 80
hidden_size = 768
batch_size = 32 #64
epoch = 20
lr = 1e-5
fixed_flag = True
seed = 2024
pretrain_path = './pretrain_model'