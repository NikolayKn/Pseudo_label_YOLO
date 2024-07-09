import os
import gdown
import shutil
import yaml
import sys
import zipfile
from ultralytics import YOLO
import shutil
import yaml
import pickle
import random
import re

# Чтение конфигурационного файла
def read_yaml_file(file_path):
    with open(file_path, 'r') as file:
        try:
            data = yaml.safe_load(file)
            return data
        except yaml.YAMLError as e:
            print("Error reading YAML file:", e)

# Подгрузить разметку в зависимости от эксперимента
def prepare_labels(
    dataset_name, # Название датасета
    train_postfix=None, # Разметка, по умолчанию оригинальная [orig, GD]
    val_postfix=None, # Разметка, по умолчанию оригинальная [orig, GD]
    test_postfix=None, # Разметка, по умолчанию оригинальная [orig, GD]
    seed= None,
    number_of_samples=None,
    data_split = None
    ):
    # Перезаписать разметку на разметку на выбранную
    for sample, postfix in zip(['train', 'val', 'test'], [train_postfix, val_postfix, test_postfix]):
        if postfix is None:
            postfix = 'orig'

        img_source = f'datasets/{dataset_name}/images/{sample}'
        lbs_dest = f'datasets/{dataset_name}/labels/{sample}'
        lbs_source = f'datasets/{dataset_name}/labels/{sample}_{postfix}'
        # !rm -rf {lbs_dest}


        shutil.rmtree(lbs_dest) # Delete directory


        if seed is not None and number_of_samples is not None and data_split is not None and sample == 'train':
            os.makedirs(lbs_dest) # Создать пустую папку

            # Получить список картинок
            image_names = sorted([x.split('.')[0] for x in os.listdir(img_source)])
            label_names = [f'{lbs_source}/{index}.txt' for index in image_names]
            random.Random(seed).shuffle(label_names)
            label_names = label_names[(data_split -1)*number_of_samples : data_split*number_of_samples ]

            for label_file in label_names:
                shutil.copy(label_file, lbs_dest)

        else:
            shutil.copytree(lbs_source, lbs_dest)

        try:
            return label_names
        except NameError:
            return None
        
def run_test(
    weight_path,
    exp,
    dir_new_test
    ):
    model =YOLO(weight_path)
    validation_results = model.val(**exp['test_config'])
    pickle.dump(validation_results.box, file = open(f'{str(validation_results.save_dir)}/results.pickle', "wb"))

    directory_old_test = str(validation_results.save_dir)

    index_save = 0
    try:
        shutil.move(directory_old_test, dir_new_test)
    except OSError:
        index_save +=1
        dir_new_test = dir_new_test + f'_{index_save}'
        shutil.move(directory_old_test, dir_new_test)


def main():
    # Path to download config file
    experiments = read_yaml_file(sys.argv[1])
    for exp in experiments:
        label_names = prepare_labels(**exp['data_config'])
        model = YOLO(exp['model'])
        results = model.train(**exp['train_config'])

        directory_old_train = str(results.save_dir)
        dir_new_train = f'{exp["saving_directory"]}/train'

        index_save = 0
        try:
            shutil.move(directory_old_train, dir_new_train)
        except OSError:
            index_save +=1
            dir_new_train = dir_new_train + f'_{index_save}'
            shutil.move(directory_old_train, dir_new_train)


        # Сохранить список файлов для обучения
        if label_names is not None:
            label_names = [x + '\n' for x in label_names]
            with open (dir_new_train + '/list_of_training_files.txt', 'w+') as f:
                f.writelines(label_names)

        #test
        weight_path = f'{dir_new_train}/weights/'
        all_weights = os.listdir(weight_path)

        if len(all_weights) > 2:
            for weight in all_weights:
                search_res = re.search('epoch(\d+).pt', weight)
                if search_res is None:
                    continue
                epoch_index = search_res.group(1)
                weight_path_full = f'{weight_path}/{weight}'
                dir_new_test = f'{exp["saving_directory"]}/test_epoch_{epoch_index}'
                run_test(weight_path_full, exp, dir_new_test)
            
            # best
            weight_path_full = f'{weight_path}/best.pt'
            dir_new_test = f'{exp["saving_directory"]}/test_best'
            run_test(weight_path_full, exp, dir_new_test)

            # last
            weight_path_full = f'{weight_path}/last.pt'
            epoch_index = exp['train_config']['epochs']
            dir_new_test = f'{exp["saving_directory"]}/test_epoch_{epoch_index}'
            run_test(weight_path_full, exp, dir_new_test)

        else:
            # best
            weight_path_full = f'{weight_path}/best.pt'
            dir_new_test = f'{exp["saving_directory"]}/test'
            run_test(weight_path_full, exp, dir_new_test)



if __name__ == "__main__":
    main()