from argparse import ArgumentParser

def train_parameters():
    parser = ArgumentParser(description='中草药分类')
    parser.add_argument('--target_path', default='D:\\PyCharm\\MedicineClassification\\data\\')
    parser.add_argument('--train_list_path', default='D:\\PyCharm\\MedicineClassification\\data\\train.txt')
    parser.add_argument('--eval_list_path', default='D:\\PyCharm\\MedicineClassification\\data\\eval.txt')
    parser.add_argument('--readme_path', default='D:\\PyCharm\\MedicineClassification\\data\\readme.json')
    parser.add_argument('--input_size', default=[3, 224, 224])
    parser.add_argument('--class_dim', default=5)
    parser.add_argument('--label_dict', default={})
    parser.add_argument('--num_epochs', default=45) #VGG25  35  45 55
    parser.add_argument('--train_batch_size', default=8)
    parser.add_argument('--skip_steps', default=50)  #10
    parser.add_argument('--save_steps', default=100) #30
    parser.add_argument('--lr', default=0.001)
    parser.add_argument('--checkpoints', default='D:\\PyCharm\\MedicineClassification\\ckpts')
    return parser.parse_args()
