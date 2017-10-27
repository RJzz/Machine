import numpy as np
import operator
import matplotlib
import matplotlib.pyplot as plt


def create_date_set():
    group = np.array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels


'''
:parameter
输入向量:in_x, 输入的训练样本：data_set, 标签向量:labels, 
表示用于选择最近邻居的数目
'''


def classify0(in_x, data_set, labels, k):
    data_set_size = data_set.shape[0]
    # tile(original, (a, b)) 将原来的矩阵行复制倍,列复制a倍
    # 计算欧氏距离
    diff_mat = np.tile(in_x, (data_set_size, 1)) - data_set
    sq_diff_mat = diff_mat ** 2
    # 相加为一个列向量
    sq_distances = sq_diff_mat.sum(axis=1)
    # 开方
    distances = sq_distances ** 0.5
    # 从小到大排列，返回该值在原来值中的索引
    sorted_dist_indices = distances.argsort()
    class_count = {}
    # 计算在邻居中哪一类最多
    for i in range(k):
        votel_label = labels[sorted_dist_indices[i]]
        class_count[votel_label] = class_count.get(votel_label, 0) + 1
    sorted_class_count = sorted(class_count.items(), key=operator.itemgetter(1), reverse=True)  #
    return sorted_class_count[0][0]


# 读取文件，形成数据集和标签
def file2matrix(filename):
    with open(filename, 'r', encoding='UTF-8') as fr:
        lines = fr.readlines()
        number_of_lines = len(lines)
        mat = np.zeros((number_of_lines, 3))
        class_label_vector = []
        index = 0
        for line in lines:
            line = line.strip()
            content = line.split('\t')
            mat[index, :] = content[0:3]
            class_label_vector.append(int(content[-1]))
            index += 1
        return mat, class_label_vector


# 归一化特征值
def auto_norm(data_set):
    min_value = data_set.min(0)
    max_value = data_set.max(0)
    ranges = max_value - min_value
    norm_data_set = np.zeros(np.shape(data_set))
    m = data_set.shape[0]
    norm_data_set = data_set - np.tile(min_value, (m, 1))
    norm_data_set = norm_data_set / np.tile(ranges, (m, 1))
    return norm_data_set, ranges, min_value


# 测试
def dating_class_test():
    ho_ratio = 0.2
    dating_data_mat, dating_labels = file2matrix("./MLiA_SourceCode/machinelearninginaction/Ch02"
                                                 "/datingTestSet2.txt")
    nor_mat, ranges, min_vals = auto_norm(dating_data_mat)
    m = nor_mat.shape[0]
    num_test_vecs = int(m * ho_ratio)
    error_count = 0.0
    for i in range(num_test_vecs):
        classifier_result = classify0(nor_mat[i, :], nor_mat[num_test_vecs:m, :],
                                      dating_labels[num_test_vecs:m], 3)
        print("the classifier came back with: %d, the real answer is: %d"
              % (classifier_result, dating_labels[i]))
        if classifier_result != dating_labels[i]:
            error_count += 1
    print("the total error rate is: %f" % (error_count / float(num_test_vecs)))


# 约会网站预测函数
def classify_person():
    result_list = ['not at all', 'in small doses', 'in large doses']
    percent_tats = float(input("percentage of time spent playing video games?"))
    ice_cream = float(input("liters of ice cream consumed per year?"))
    ff_miles = float(input("frequent flier miles earned per year?"))

    dating_data_mat, dating_labels = file2matrix("./MLiA_SourceCode/machinelearninginaction/Ch02"
                                                 "/datingTestSet2.txt")
    nor_mat, ranges, min_vals = auto_norm(dating_data_mat)
    in_arr = np.array([ff_miles, percent_tats, ice_cream])
    classifier_result = classify0((in_arr - min_vals) / ranges, nor_mat, dating_labels, 3)

    print("You will probably like this person: ", result_list[classifier_result - 1])


if __name__ == "__main__":
    # group, labels = create_date_set()
    # print(classify0([0, 0], group, labels, 3))
    # dating_data_mat, dating_labels = file2matrix("./MLiA_SourceCode/machinelearninginaction/Ch02"
    #                                              "/datingTestSet2.txt")
    # print(dating_data_mat, dating_labels)
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # ax.scatter(dating_data_mat[:, 1], dating_data_mat[:, 2],
    #            15.0 * np.array(dating_labels), 15.0 * np.array(dating_labels))
    # plt.show()
    dating_class_test()
    classify_person()
