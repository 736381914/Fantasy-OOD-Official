# 提取places365_test_list中的子集
import os

# 假设txt文件和jpg图像都在同一个目录下
directory = '/home/zrf/datasets/small_OOD_dataset/places365/test_subset'  # 替换为你的图像和txt文件所在的目录
txt_file = '/home/zrf/datasets/small_OOD_dataset/places365_test_list.txt'  # 替换为你的txt文件名

# 读取txt文件中的图像名称
with open(txt_file, 'r') as file:
    image_names = [line.strip() for line in file.readlines()]
print(image_names,len(image_names))
# 遍历目录中的所有文件
for filename in os.listdir(directory):
    if filename.endswith(".jpg") and filename not in image_names:
        # 如果文件不在txt文件的列表中，删除它
        os.remove(os.path.join(directory, filename))
        print(f"Deleted {filename}")
    elif filename.endswith(".jpg"):
        # 如果文件在列表中，保留它
        print(f"Kept {filename}")