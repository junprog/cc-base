import os
import json
import glob
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Create datset path json file')
    parser.add_argument(
        '--data-dir',
        default='',
        help='dataset directory ucf-qnrf: /UCF-QNRF_ECCV18, shanghai-tech-part*: /ShanghaiTech/part_*、ShanghaiTechRGBD: /ShanghaiTechRGBD'
    )
    parser.add_argument(
        '--val-rate',
        default=4,
        type=int,
        help='ratio of num of validation datas (e.g. 4 -> 3/4: tr, 1/4: vl)'
    )
    args = parser.parse_args()
    return args

def create_json():
    # path to folder that contains images
    args = parse_args()
    data_dir = args.data_dir        ### Datasetのディレクトリ
    val_rate = args.val_rate        ### validation datasの割合

    train_json_name = 'train.json'
    val_json_name = 'val.json'
    test_json_name = 'test.json'
    img_format = '*.jpg'

    if 'rescale' in os.path.basename(data_dir):
        if 'UCF-QNRF' in os.path.basename(data_dir):
            dataset_name = 'rescale-ucf-qnrf'
            train_img_dir = os.path.join(data_dir, 'Train')
            test_img_dir = os.path.join(data_dir, 'Test')
        elif 'part_A' in os.path.basename(data_dir):
            dataset_name = 'rescale-shanghai-tech-a'
            train_img_dir = os.path.join(data_dir, 'train_data/images')
            test_img_dir = os.path.join(data_dir, 'test_data/images')
    else:
        if 'UCF-QNRF' in os.path.basename(data_dir):
            dataset_name = 'ucf-qnrf'
            train_img_dir = os.path.join(data_dir, 'Train')
            test_img_dir = os.path.join(data_dir, 'Test')
        elif 'part_A' in os.path.basename(data_dir):
            dataset_name = 'shanghai-tech-a'
            train_img_dir = os.path.join(data_dir, 'train_data/images')
            test_img_dir = os.path.join(data_dir, 'test_data/images')
        elif 'part_B' in os.path.basename(data_dir):
            dataset_name = 'shanghai-tech-b'
            train_img_dir = os.path.join(data_dir, 'train_data/images')
            test_img_dir = os.path.join(data_dir, 'test_data/images')
        elif 'RGBD' in os.path.basename(data_dir):
            dataset_name = 'shanghai-tech-rgbd'
            train_img_dir = os.path.join(data_dir, 'train_data/train_img')
            test_img_dir = os.path.join(data_dir, 'test_data/test_img')
            img_format = '*.png'
        elif 'synthetic-datas' == os.path.basename(data_dir):
            dataset_name = 'synthetic-dataset'
            train_img_dir = os.path.join(data_dir, 'train')
            test_img_dir = os.path.join(data_dir, 'test')
            img_format = '*.png'
        elif 'synthetic-datas-v2' == os.path.basename(data_dir):
            dataset_name = 'synthetic-dataset-v2'
            train_img_dir = os.path.join(data_dir, 'train')
            test_img_dir = os.path.join(data_dir, 'test')
            img_format = '*.png'
        elif 'synthetic-datas-2d' == os.path.basename(data_dir):
            dataset_name = 'synthetic-dataset-2d'
            train_img_dir = os.path.join(data_dir, 'train')
            test_img_dir = os.path.join(data_dir, 'test')
        elif 'synthetic-datas-2d-bg' == os.path.basename(data_dir):
            dataset_name = 'synthetic-dataset-2d-bg'
            train_img_dir = os.path.join(data_dir, 'train')
            test_img_dir = os.path.join(data_dir, 'test')
        

    ## 現在のディレクトリにjson/を作成
    if os.path.isdir('json') == False:
        os.mkdir('json')

    ### json/ディレクトリにdataset_name/を作成
    data_json_dir = os.path.join('json', dataset_name)
    if os.path.isdir(data_json_dir) == False:
        os.mkdir(data_json_dir)

    train_json_path = os.path.join(data_json_dir, train_json_name)
    val_json_path = os.path.join(data_json_dir, val_json_name)  

    ### train, val json file 作成
    if os.path.exists(train_json_path) or os.path.exists(val_json_path):
        print("{} dataset's tr val json file is already exists.".format(dataset_name))
    else:
        if val_rate == 0: # validationなしの場合 (default : 4)
            img_list_tr = []
            for i, img_path in enumerate(glob.glob(os.path.join(train_img_dir, img_format))):
                img_list_tr.append(img_path)

            with open(train_json_path, 'w') as f:
                json.dump(img_list_tr,f)

        else:  # validationありの場合 (default : 4)
            img_list_tr = []
            img_list_vl = []

            for i, img_path in enumerate(glob.glob(os.path.join(train_img_dir, img_format))):
                if i % val_rate == 0: # validation ratio
                    img_list_vl.append(img_path)
                else:
                    img_list_tr.append(img_path)

            with open(train_json_path, 'w') as f:
                json.dump(img_list_tr, f)

            with open(val_json_path, 'w') as f:
                json.dump(img_list_vl, f)


    ### test json file 作成
    test_json_path = os.path.join(data_json_dir, test_json_name)

    if os.path.exists(test_json_path):
        print("{} dataset's test json file is already exists.".format(dataset_name))
    else:
        img_list_test = []

        for i, img_path in enumerate(glob.glob(os.path.join(test_img_dir, img_format))):
            img_list_test.append(img_path)

        with open(test_json_path, 'w') as f:
            json.dump(img_list_test, f)

if __name__ == '__main__':
    create_json()