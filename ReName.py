import os

def reName2(path):
    """
    重命名函数fun2
    输入：文件夹路径
    功能：对某一个文件夹中的某一类文件进行统一命名，命名格式为：基础名+数字序号
    """
    i = 1
    suffix = '.jpg'  # 设置后缀，筛选特定文件以更改名称
    for file in os.listdir(path):
        if file.endswith(suffix):
            if os.path.isfile(os.path.join(path, file)):
                new_name = file.replace(file, "huaihua_%d" % i + suffix)  # 根据需要设置基本文件名
                os.rename(os.path.join(path, file), os.path.join(path, new_name))
                i += 1
    print("End")



if __name__ =='__main__':
    filePath = r'D:\PyCharm\MedicineClassification\data\ChineseMedicine\huaihua'
    reName2(filePath)