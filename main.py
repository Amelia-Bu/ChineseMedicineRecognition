import utils
from torch.utils.data import DataLoader
import trainPara
import model
import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np


if __name__ == '__main__':

    params = trainPara.train_parameters()

    with open(params.train_list_path, 'w') as f:
        f.seek(0)
        f.truncate()
    with open(params.eval_list_path, 'w') as f:
        f.seek(0)
        f.truncate()

        # 生成数据列表
    utils.get_data_list(params.target_path, params.train_list_path, params.eval_list_path)

    # 训练数据加载
    train_dataset = utils.dataset('D:\\PyCharm\\MedicineClassification\\data\\', mode='train')
    train_loader = DataLoader(train_dataset, batch_size=params.train_batch_size, shuffle=True)

    # 测试数据加载
    eval_dataset = utils.dataset('D:\\PyCharm\\MedicineClassification\\data\\', mode='eval')
    eval_loader = DataLoader(eval_dataset, batch_size=8, shuffle=False)


    '''
    参数初始化
    '''
    target_path = params.target_path
    train_list_path = params.train_list_path
    eval_list_path = params.eval_list_path




    model = model.VGGNet().cuda()
    # model = model.AlexNet().cuda()
    model.train()
    print(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=utils.train_parameters['lr'])

    steps = 0
    Iters, total_loss, total_acc = [], [], []
    epos = []

    for epo in range(params.num_epochs):
        epos.append(epo)
        for _, data in enumerate(train_loader):
            steps += 1
            x_data = data[0].cuda()
            y_data = data[1].cuda()
            predicts, acc = model(x_data, y_data)
            y_data = torch.flatten(y_data)
            loss = F.cross_entropy(predicts, y_data)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            if steps % utils.train_parameters["skip_steps"] == 0:
                Iters.append(steps)
                # total_loss.append(loss)
                # total_acc.append(acc)
                # 打印中间过程
                # print('epo: {}, step: {}, loss is: {}, acc is: {}' \
                #       .format(epo, steps, loss.detach().numpy(), acc))
                print('epo: {}, step: {}, loss is: {}, acc is: {}' \
                      .format(epo, steps, loss, acc))
            # 保存模型参数
            if steps % params.save_steps == 0:
                save_path = params.checkpoints + "\\" + "save_dir_" + str(steps) + '.pdparams'
                print('save model to: ' + save_path)
                torch.save(model.state_dict(), save_path)
        total_loss.append(loss)
        total_acc.append(acc)

    torch.save(model.state_dict(), params.checkpoints + "\\" + "save_dir_final.pdparams")
    total_loss = [_.clone().cpu().detach().requires_grad_(False) for _ in total_loss]
    toral_loss = [_.numpy() for _ in total_loss]
    # utils.draw_process("trainning loss", "red", Iters, total_loss, "trainning loss")
    # utils.draw_process("trainning acc", "green", Iters, total_acc, "trainning acc")
    utils.draw_process("trainning loss", "red", epos, total_loss, "trainning loss")
    utils.draw_process("trainning acc", "green", epos, total_acc, "trainning acc")

    '''
    模型评估
    '''
    model_state_dict = torch.load('D:\\PyCharm\\MedicineClassification\\ckpts\\save_dir_final.pdparams')
    # model_eval = model.VGGNet().cuda()
    # model_eval = model.AlexNet().cuda()
    model.load_state_dict(model_state_dict)
    model.eval()
    accs = []

    for _, data in enumerate(eval_loader):
        x_data = data[0].cuda()
        y_data = data[1].cuda()
        predicts = model(x_data)
        _, predicts = torch.max(predicts.data, 1)
        y_data = y_data.squeeze()
        acc = (predicts == y_data).sum().item() / len(y_data)
        accs.append(acc)
    print('模型在验证集上的准确率为：', np.mean(accs))