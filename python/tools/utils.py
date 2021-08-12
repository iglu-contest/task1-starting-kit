import json
import numpy as np
import os

def calc_single_xml(file_path=None):
    result = {}

    with open(file_path, 'r') as f:
        data = f.readlines()
    for line in data:
        color = line.strip().split('_')[-2]
        if color in result:
            result[color]+=1
        else:
            result[color]=1
    return result, len(result), len(data)
    

def calc_block_color_number(dir_name=None):
    # for subdir, dirs, files in os.walk(dir_name):
    #         for file in files:
    #             #print os.path.join(subdir, file)
    #             filepath = subdir + os.sep + file
    product_list = []
    easy, medium, hard = [], [], []
    for i in range(1, 158):
        filepath = dir_name + '/C{}.xml'.format(i)
        print("file name: {}".format(filepath))
        result, color_num, block_num = calc_single_xml(filepath)
        print(result, ' color num: ',color_num, ' block num: ', block_num, 'product: ', color_num*block_num)
        product_list.append(color_num*block_num)
        if product_list[-1]<=50:
            # easy.append('C{}.xml'.format(i))
            easy.append(i)
        elif product_list[-1]<=150:
            # medium.append('C{}.xml'.format(i))
            medium.append(i)
        else:
            # hard.append('C{}.xml'.format(i))
            hard.append(i)

        # print(color_num, block_num)
    # print(product_list)
    # a=sorted(product_list)
    # print(a)
    print('easy: ', easy, len(easy))
    print('medium: ', medium, len(medium))
    print('hard: ', hard, len(hard))



if __name__ == '__main__':
    dir_name = './data/gold-configurations'
    calc_block_color_number(dir_name)
                
