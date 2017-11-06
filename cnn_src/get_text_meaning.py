import os
import pandas as pd
import matplotlib.pyplot as plt

if __name__ == '__main__':

    folder_path = '..\文档'
    folder_list = os.listdir(folder_path)

    length = {}
    for folder in folder_list:
        new_folder_path = os.path.join(folder_path, folder)
        file_list = os.listdir(new_folder_path)

        SUM = 0
        for file in file_list:
            file_name = os.path.join(new_folder_path, file)
            with open(os.path.join(new_folder_path, file), 'r', encoding='gbk') as fp:
                try:
                    context = fp.read()
                    if str(len(context)) not in length:
                        length[len(context)] = 1
                    else:
                        length[len(context)] += 1
                except:
                    fp.close()
                    SUM += 1
                    os.system("rm " + file_name)

        print(new_folder_path, "sum = ", SUM)

    df = pd.DataFrame(length)
    print(df)

