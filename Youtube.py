# -*- coding:UTF-8 -*-

import csv
import pandas as pd
from pytube import YouTube
import os

from pytube.exceptions import RegexMatchError

if __name__ == '__main__':
    df = pd.read_csv('E:\\video\\MSVD\\MSR Video Description Corpus.csv')
    print('type of df', type(df))
    value = df.values
    value = value.tolist()
    print(value[-1])
    value = [x[0] for x in value]
    videos = list(set(value))
    videos.sort(key=value.index) # 保证排序不变，实际上无所谓排序不排序
    # print('len:', len(value[0]), value[0])

    un_down_load = []
    age_or_something_else = []
    if (not os.path.exists('./Video')):
        os.makedirs('./Video')
    for video in videos[2:]:
        # 发现某些视频在youtube上已经不存在了
        if(not os.path.exists('./Video/' + str(video) + '.mp4')):
            try:
                yt = YouTube("http://www.youtube.com/watch?v=" + str(video))
                print_out = yt.streams.filter(subtype='mp4').first()
                # print_out.set_filename(str(video))
                try:
                    print_out.download(output_path='./Video', file_name=str(video) + '.mp4')
                except RegexMatchError:
                    print(str(video) + ' not exits')
                    un_down_load.append(str(video))
            except Exception:
                print('there is something wrong happening，emmmmmmmmmm ', str(video))
                age_or_something_else.append(str(video))
        else:
            age_or_something_else.append(str(video) + '\n')


    with open('./un_down_load.txt', 'w', encoding='utf-8') as input:
        input.writelines(un_down_load)

    with open('./age_or_something_else.txt', 'w', encoding='utf-8') as input:
        input.writelines(age_or_something_else)







        # print_out = yt.streams.filter(subtype='mp4').first()
        # # print_out.set_filename(str(video))
        # print_out.download(output_path='./Video', file_name=str(video) + '.mp4')





    # print_out.first()
    # print(print_out)
    # print_out[0].download()
    # print(yt.streams.all())
    # with open('mv89psg6zh4.mp4', 'w', encoding='utf-8') as input:
    #     input.write(print_out.first())
    # print_out.first()

    # print_out = str(yt.filter('mp4')[-1])
    # print




