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
from datetime import datetime 

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
    seed=None,
    number_samples=None
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


        if seed is not None and number_samples is not None and sample == 'train':
            os.makedirs(lbs_dest) # Создать пустую папку

            # Получить список картинок
            image_names = sorted([x.split('.')[0] for x in os.listdir(img_source)])
            label_names = [f'{lbs_source}/{index}.txt' for index in image_names]
            random.Random(seed).shuffle(label_names)

            label_names = label_names[:number_samples]

            for label_file in label_names:
                shutil.copy(label_file, lbs_dest)

        else:
            shutil.copytree(lbs_source, lbs_dest)

        try:
            return label_names
        except NameError:
            return None


def main():
    # Path to download config file
    experiments = read_yaml_file(sys.argv[1])
    for exp in experiments:
        label_names = prepare_labels(**exp['data_config'])
        model = YOLO(exp['model'])
        results = model.train(**exp['train_config'])

        directory_old_train = str(results.save_dir)
        dir_new_train = f'{exp["saving_directory"]}_{exp["model"].split(".")[0]}_train'
        try:
            os.rename(directory_old_train, dir_new_train)
        except OSError:
            now = datetime.now().strftime("%H_%M_%S") 
            dir_new_train = dir_new_train + str(now)
            os.rename(directory_old_train, dir_new_train)


        # Сохранить список файлов для обучения
        if label_names is not None:
            label_names = [x + '\n' for x in label_names]
            with open (dir_new_train + '/list_of_training_files.txt', 'w+') as f:
                f.writelines(label_names)

        #test
        weight_path = f'{dir_new_train}/weights/best.pt'
        model =YOLO(weight_path)
        validation_results = model.val(**exp['test_config'])
        pickle.dump(validation_results.box, file = open(f'{str(validation_results.save_dir)}/results.pickle', "wb"))

        directory_old_test = str(validation_results.save_dir)
        dir_new_test = f'{exp["saving_directory"]}_{exp["model"].split(".")[0]}_test'

        try:
            os.rename(directory_old_test, dir_new_test)
        except OSError:
            now = datetime.now().strftime("%H_%M_%S") 
            dir_new_test = dir_new_test + str(now)
            os.rename(directory_old_test, dir_new_test)



if __name__ == "__main__":
    main()