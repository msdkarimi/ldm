import json
import pandas as pd
import os

def convert(json_file_path, group_by_key='image_id', creat_list_of='caption', save=True):
    with open(json_file_path) as json_file:
        _data = json.load(json_file)

    chunk =json_file_path.split(os.path.sep)[-1].split('.')[0]
    _dataset =  _data['annotations']
    _df = pd.DataFrame.from_dict(_dataset)
    _df = _df.groupby(group_by_key)[creat_list_of].apply(list).reset_index()
    if save:
        _df.to_csv(f'{chunk}.txt', index=False, header=False, sep='|')
    else:
        print(_df.head())


if __name__ == '__main__':
    try:
        convert('validation.json')
        exit(0)
    except Exception as e:
        print(e)
        exit(1)


