import os
import csv

kitti_label_file_path = "../KITTI_DATASET_ROOT/training/label_2/"
file_list = [file_name for file_name in os.listdir(kitti_label_file_path)]
file_list.sort()


sentence_mapping_id = {}
use_class = ["car", "van", "truck"]
sentence_one_class_one_quantity_list = ["There is 1 " + i + " in the image." for i in use_class]
sentence_one_class_many_quantity_list = ["There are " + str(j) + " " + i + "s" + " in the image." for i in use_class for j in range(2, 11)]
sentence_two_class_one_quantity_one_quantity_list = ["There are 1 " + use_class[i] + " and 1 " + use_class[j] + " in the image." for i in range(len(use_class)) for j in range(i + 1, len(use_class))]
sentence_two_class_one_quantity_many_quantity_list = ["There are 1 " + use_class[i] + " and " + str(k) + " " + use_class[j] + "s in the image." for i in range(len(use_class)) for j in range(i + 1, len(use_class)) for k in range(2, 11)]
sentence_two_class_many_quantity_one_quantity_list = ["There are " + str(j) + " " + use_class[i] + "s and 1 " + use_class[k] + " in the image." for i in range(len(use_class)) for j in range(2, 11) for k in range(i + 1, len(use_class))]
sentence_two_class_many_quantity_many_quantity_list = ["There are " + str(j) + " " + use_class[i] + "s and " + str(l) + " " + use_class[k] + "s in the image." for i in range(len(use_class)) for j in range(2, 11) for k in range(i + 1, len(use_class)) for l in range(2, 11)]
sentence_three_class_one_quantity_one_quantity_one_quantity_list = ["There are 1 " + use_class[0] + ", 1 " + use_class[1] + " and 1 " + use_class[2] + " in the image."]
sentence_three_class_one_quantity_one_quantity_many_quantity_list = ["There are 1 " + use_class[0] + ", 1 " + use_class[1] + " and " + str(i) + " " + use_class[2] + "s in the image." for i in range(2, 11)]
sentence_three_class_one_quantity_many_quantity_one_quantity_list = ["There are 1 " + use_class[0] + ", " + str(i) + " " + use_class[1] + "s and 1 " + use_class[2] + " in the image." for i in range(2, 11)]
sentence_three_class_many_quantity_one_quantity_one_quantity_list = ["There are " + str(i) + " " + use_class[0] + "s, 1 " + use_class[1] + " and 1 " + use_class[2] + " in the image." for i in range(2, 11)]
sentence_three_class_many_quantity_many_quantity_one_quantity_list = ["There are " + str(i) + " " + use_class[0] + "s, " + str(j) + " " + use_class[1] + "s and 1 " + use_class[2] + " in the image." for i in range(2, 11) for j in range(2, 11)]
sentence_three_class_many_quantity_one_quantity_many_quantity_list = ["There are " + str(i) + " " + use_class[0] + "s, 1 " + use_class[1] + " and " + str(j) + " " + use_class[2] + "s in the image." for i in range(2, 11) for j in range(2, 11)]
sentence_three_class_one_quantity_many_quantity_many_quantity_list = ["There are 1 " + use_class[0] + ", " + str(i) + " " + use_class[1] + "s and " + str(j) + " " + use_class[2] + "s in the image." for i in range(2, 11) for j in range(2, 11)]
sentence_three_class_many_quantity_many_quantity_many_quantity_list = ["There are " + str(i) + " " + use_class[0] + "s, " + str(j) + " " + use_class[1] + "s and " + str(k) + " " + use_class[2] + "s in the image." for i in range(2, 11) for j in range(2, 11) for k in range(2, 11)]

sentence_list = sentence_one_class_one_quantity_list + \
                sentence_one_class_many_quantity_list + \
                sentence_two_class_one_quantity_one_quantity_list + \
                sentence_two_class_one_quantity_many_quantity_list + \
                sentence_two_class_many_quantity_one_quantity_list + \
                sentence_two_class_many_quantity_many_quantity_list + \
                sentence_three_class_one_quantity_one_quantity_one_quantity_list + \
                sentence_three_class_one_quantity_one_quantity_many_quantity_list + \
                sentence_three_class_one_quantity_many_quantity_one_quantity_list + \
                sentence_three_class_many_quantity_one_quantity_one_quantity_list + \
                sentence_three_class_many_quantity_many_quantity_one_quantity_list + \
                sentence_three_class_many_quantity_one_quantity_many_quantity_list + \
                sentence_three_class_one_quantity_many_quantity_many_quantity_list + \
                sentence_three_class_many_quantity_many_quantity_many_quantity_list

for i in range(len(sentence_list)):
    sentence_mapping_id[sentence_list[i]] = i


with open("./label_2_sentence.csv", "w", newline="") as write_file:
    writer = csv.writer(write_file)
    for i in file_list:
        type_list = [["car", 0, 0], ["van", 0, 0], ["truck", 0, 0], ["pedestrian", 0, 0], ["person_sitting", 0, 0], ["cyclist", 0, 0], ["tram", 0, 0], ["misc", 0, 0]]
        with open(kitti_label_file_path + i, "r") as file:
            for line in file.readlines():
                line_list = line.split(" ")
                if line_list[0] == "Car":
                    type_list[0][2] += 1
                elif line_list[0] == "Van":
                    type_list[1][2] += 1
                elif line_list[0] == "Truck":
                    type_list[2][2] += 1
            if type_list[0][2] > 10 or type_list[1][2] > 10 or type_list[2][2] > 10:
                continue
            
            tl_num = 0
            tl_ty = []
            for tl in type_list:
                if tl[2] != 0:
                    tl_num += 1
                    if tl[2] > 1:
                        tl[1] = 1
                    tl_ty.append(tl)
            if tl_num == 0:
                sentence = "None"
                continue
            elif tl_num == 1:
                if tl_ty[0][1] == 0:
                    sentence = "There is " + str(tl_ty[0][2]) + " " + tl_ty[0][0] + " in the image."
                else:
                    sentence = "There are " + str(tl_ty[0][2]) + " " + tl_ty[0][0] + "s" + " in the image."
            else:
                sentence = "There are "
                for tl in tl_ty[:-1]:
                    if tl[1] == 0:
                        sentence = sentence + str(tl[2]) + " " + tl[0] + ", "
                    else:
                        sentence = sentence + str(tl[2]) + " " + tl[0] + "s, "
                if tl_ty[-1][1] == 0:
                    sentence = sentence[:-2] + " and " + str(tl_ty[-1][2]) + " " + tl_ty[-1][0] + " in the image."
                else:
                    sentence = sentence[:-2] + " and " + str(tl_ty[-1][2]) + " " + tl_ty[-1][0] + "s" + " in the image."
        writer.writerow([i, sentence, sentence_mapping_id[sentence]])