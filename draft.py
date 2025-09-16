import torch
import os
image_dir = r'D:\user_3188\AIC_SAR\source\train\images'
ann_dir = r'D:\user_3188\AIC_SAR\source\train\annfiles'
valid_extensions = ('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp')

line="163.0 144.0 175.0 138.0 190.0 167.0 178.0 173.0 ship 0"
parts = line.split()
print(parts)