# ChineseMedicineRecognition
Deep Learning Course Assignment （Already submitted）

网络选择的是经典的AlexNet 和 VGGNet。<br>

![image](https://github.com/Amelia-Bu/ChineseMedicineRecognition/assets/56344489/a8dd7a8f-9d9e-41f8-98e1-246969c7d83b)

![image](https://github.com/Amelia-Bu/ChineseMedicineRecognition/assets/56344489/8e06b6d0-493c-490d-bf4f-012e38b7a0ca)

# 数据集：
实验采用的数据集是飞桨平台提供的中草药识别数据集。该数据集共包含917张图片，其中训练集787张图片，验证集115张图片，10张测试图片。图片的格式为jpg格式。该数据集共包含5种类别，分别是百合、金银花、枸杞、槐花和党参。训练集的格式为这五种类别下分别包含该类别下的图片。其中百合类别下包含图片180张，金银花类别下包含图片180张，枸杞类别下包含图片185张，槐花类别下包含图片167张，党参类别下包含图片190张。<br>
![image](https://github.com/Amelia-Bu/ChineseMedicineRecognition/assets/56344489/adadcc19-74d1-40a1-ac26-5c755ed36c55)


# 数据预处理
数据处理主要进行以下几个步骤：下载原始数据集；解压数据集；对数据集进行规范命名；按照比例划分训练集与验证集；乱序生成数据列表；定义数据读取器。<br>
![image](https://github.com/Amelia-Bu/ChineseMedicineRecognition/assets/56344489/9365ec01-6e4a-44c1-9c8f-a3d33295f810)

# 模型结构
Alex net<br>
![image](https://github.com/Amelia-Bu/ChineseMedicineRecognition/assets/56344489/daaafebf-722d-4a33-9b32-842f0d066995)

VGGNet<br>
![image](https://github.com/Amelia-Bu/ChineseMedicineRecognition/assets/56344489/1437c64c-1f12-4512-96d0-d89d1f88a897)
![image](https://github.com/Amelia-Bu/ChineseMedicineRecognition/assets/56344489/2e6c0ac1-89aa-4040-a7dc-1b156df172e4)

# 实验结果
## AlexNet
使用AlexNet模型在训练集上的训练损失与精度结果展示<br>
![image](https://github.com/Amelia-Bu/ChineseMedicineRecognition/assets/56344489/5fcf10ed-a30f-4f11-ab11-3c905875c738)<br>
预测结果。使用AlexNet在测试集上进行测试的结果如下图所示。可以观察到使用AlexNet进行训练后可以在测试集上达到不错的预测结果。<br>
![image](https://github.com/Amelia-Bu/ChineseMedicineRecognition/assets/56344489/8dd3758e-f601-4aaf-808d-f50dc705aa9d)<br>

## VGGNet
使用VGGNet模型在训练集上的训练损失与精度结果展示<br>
![image](https://github.com/Amelia-Bu/ChineseMedicineRecognition/assets/56344489/e58189cf-7721-484c-b970-6495d1305506)<br>
使用VGGNet在测试集上进行测试的结果如下图16所示。可以观察到使用VGGNet进行训练后可以在测试集上的预测结果并没有很好，说明模型的超参数还有待优化，但是受限于计算资源，没能做出更好的调整。<br>
![image](https://github.com/Amelia-Bu/ChineseMedicineRecognition/assets/56344489/01c4ae49-4ac5-4b0d-b393-efe6da381a9c)<br>

# 参数调整
在配置好模型进行训练后，模型的精度往往达不到我们的要求。这时候就需要进行调参。以VGGNet模型为例，通过调整训练epoch数可以使训练损失缓慢收敛。但是在调整过程中，验证集的精度从0.66提高到0.69，然后又下降到0.64。模型的训练损失逐渐收敛但是验证集精度却在下降。经过分析认为可能是出现了过拟合现象。<br>
![image](https://github.com/Amelia-Bu/ChineseMedicineRecognition/assets/56344489/47c1d0db-8345-4fd8-8f5b-c8567e9daf34)<br>

# 改进方法
使用AlexNet进行分类预测时，经过调整超参数使得分类精度从50%~60%实现了提升。但是实验在验证集上的分类精度普遍在75%~80%之间，还有近一步的提升空间。之后可以通过一些其他的调参方法进行参数调整使得验证精度进一步提高。与VGGNet进行对比发现，改进模型也会提升分类精度。所以还可以通过进一步调整模型达到更好的分类效果<br>
