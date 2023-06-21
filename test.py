import numpy as np
from PIL import Image
import trainPara
import torch
import model
import os
import utils

if __name__ == '__main__':

    params = trainPara.train_parameters()
    utils.get_data_list(params.target_path, params.train_list_path, params.eval_list_path)
    label_dic = utils.train_parameters['label_dict']
    '''这里是test哟'''
    test_dir = 'D:\\PyCharm\\MedicineClassification\\data\\ChineseMedicineInfer'
    model_state_dict = torch.load('D:\\PyCharm\\MedicineClassification\\ckpts\\save_dir_final.pdparams')
    # model_state_dict = torch.load('D:\\PyCharm\\MedicineClassification\\ckpts\\save_dir_270.pdparams')
    model_predict = model.VGGNet().cuda()
    # model_predict = model.AlexNet().cuda()
    model_predict.load_state_dict(model_state_dict)
    model_predict.eval()
    infer_imgs_path = os.listdir('D:\\PyCharm\\MedicineClassification\\data\\ChineseMedicineInfer')


    print(infer_imgs_path)
    for infer_img_path in infer_imgs_path:
        infer_img = utils.load_image("{}\\{}".format(test_dir, infer_img_path))
        infer_img = infer_img[np.newaxis, :, :, :] # reshape(-1,3,224,224)
        infer_img = torch.tensor(infer_img).cuda()
        result = model_predict(infer_img)

        lab = np.argmax(result.cpu().detach().numpy())

        print("样本: {},被预测为:{}".format(infer_img_path, label_dic[str(lab)]))

