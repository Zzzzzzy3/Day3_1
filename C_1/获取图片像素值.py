import cv2

# 读取图片
image_path = "D:/user_3188/AIC_SAR/Day3_1/B_1/image/000001.png"
img = cv2.imread(image_path)

# 获取图片尺寸
height, width, channels = img.shape
print(f"height:{height}, width:{width}, channels:{channels}")
# 获取特定位置的像素值(BGR格式)
x, y = 100, 100
b, g, r = img[y, x]
print(f"在位置({x},{y}): B={b}, G={g}, R={r}")

# 获取所有像素值
# 注意：OpenCV使用BGR格式而不是RGB