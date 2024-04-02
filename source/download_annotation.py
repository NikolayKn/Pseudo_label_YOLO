import os
import gdown
import shutil
import yaml
import sys
import zipfile

# Чтение конфигурационного файла
def read_yaml_file(file_path):
    with open(file_path, 'r') as file:
        try:
            data = yaml.safe_load(file)
            return data
        except yaml.YAMLError as e:
            print("Error reading YAML file:", e)

# Загрузить разметку с гугл диска
def load_pseudo_labels(
        dataset_name, #VOC
        ann_name, # Название разметки
        ann_postfix, # Постфикс разметки
        url # Урл, чтобы скачать (google disk)
        ):
    
    save_dir = f'data/pseudo_labels/{dataset_name}'
    os.makedirs(save_dir, exist_ok=True)
    # Загрузка разметки
    filename = f'{save_dir}/{ann_name}.zip'
    gdown.download(url, filename, quiet = False, fuzzy = True)
    # !unzip -d {save_dir} {filename}
    with zipfile.ZipFile(filename, 'r') as zip_ref:
        zip_ref.extractall(save_dir) 

    # Копирование в папку с датасетом
    for sample in (['train', 'test', 'val']):
        lbs_source = f'{save_dir}/{ann_name}/{sample}'
        if os.path.exists(lbs_source) is False:
            print(f'{lbs_source} does not exisist')
            continue
        lbs_dest = f'datasets/{dataset_name}/labels/{sample}_{ann_postfix}'
        shutil.copytree(lbs_source, lbs_dest)


def main():
    # Path to download config file
    config = read_yaml_file(sys.argv[1])
    load_pseudo_labels(
        dataset_name=config['dataset_name'], #VOC
        ann_name=config['ann_name'], # Название разметки
        ann_postfix=config['ann_postfix'], # Постфикс разметки
        url=config['url'] # Урл, чтобы скачать (google disk)
        )

if __name__ == "__main__":
    main()