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

# Чтение конфигурационного файла
def read_yaml_file(file_path):
    with open(file_path, 'r') as file:
        try:
            data = yaml.safe_load(file)
            return data
        except yaml.YAMLError as e:
            print("Error reading YAML file:", e)

# Подгрузить разметку в зависимости от эксперимента
def prepare_labels(dataset_name, # Название датасета
                   train_postfix=None, # Разметка, по умолчанию оригинальная [orig, GD]
                   val_postfix=None, # Разметка, по умолчанию оригинальная [orig, GD]
                   test_postfix=None # Разметка, по умолчанию оригинальная [orig, GD]
                   ):
    # Перезаписать разметку на разметку на выбранную
    for sample, postfix in zip(['train', 'val', 'test'], [train_postfix, val_postfix, test_postfix]):
        if postfix is None:
            postfix = 'orig'

        lbs_dest = f'datasets/{dataset_name}/labels/{sample}'
        lbs_source = f'datasets/{dataset_name}/labels/{sample}_{postfix}'
        # !rm -rf {lbs_dest}
        shutil.rmtree(lbs_dest)
        shutil.copytree(lbs_source, lbs_dest)


def main():
    # Path to download config file
    experiments = read_yaml_file(sys.argv[1])
    for exp in experiments:
        prepare_labels(**exp['data_config'])
        model = YOLO(exp['model'])
        results = model.train(**exp['train_config'])

        directory_old_train = str(results.save_dir)
        dir_new_train = f'{exp["saving_directory"]}_{exp["model"].split(".")[0]}_train'
        os.rename(directory_old_train, dir_new_train)

        #test
        weight_path = f'{dir_new_train}/weights/best.pt'
        model =YOLO(weight_path)
        validation_results = model.val(**exp['test_config'])
        pickle.dump(validation_results.box, file = open(f'{str(validation_results.save_dir)}/results.pickle', "wb"))

        directory_old_test = str(validation_results.save_dir)
        dir_new_test = f'{exp["saving_directory"]}_{exp["model"].split(".")[0]}_test'
        os.rename(directory_old_test, dir_new_test)


if __name__ == "__main__":
    main()