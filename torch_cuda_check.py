

import torch
print(torch.cuda.is_available())      # True 表示 GPU 可用
print(torch.cuda.device_count())      # 顯示 GPU 數量
print(torch.cuda.get_device_name(0))  # 顯示第 0 張 GPU 名稱
