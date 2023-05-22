import os
import csv

kitti_label_file_path = "../kitti/training/label_2/"
file_list = [file_name for file_name in os.listdir(kitti_label_file_path)]
file_list.sort()

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
        writer.writerow([i, sentence])