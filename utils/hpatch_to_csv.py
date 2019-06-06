# -*- coding: utf-8 -*-
# @Time    : 2018-8-4 10:43
# @Author  : xylon

"""
produce csv file for HPatches dataset
"""

import os
import csv

hpatch_root_dir = ""  # data directory for HPatches dataset
csv_name = ""  # csv file name you want to save

hpatch_folder_list = os.listdir(hpatch_root_dir)

with open(os.path.join(hpatch_root_dir, "..", csv_name), "w", newline="") as csvfile:
    writer = csv.writer(csvfile)

    writer.writerow(
        ["folder", "im1", "im2", "h1", "h2", "h3", "h4", "h5", "h6", "h7", "h8", "h9"]
    )
    for f in hpatch_folder_list:
        for i in range(2, 7):
            homo_file = open(
                os.path.join(hpatch_root_dir, f, "H_1_" + str(i)),
                encoding="utf-8",
                mode="r",
            )
            h1 = homo_file.readline().strip("\n").split(" ")
            h2 = homo_file.readline().strip("\n").split(" ")
            h3 = homo_file.readline().strip("\n").split(" ")
            writer.writerow(
                [
                    f,
                    "1.ppm",
                    str(i) + ".ppm",
                    h1[0],
                    h1[1],
                    h1[2],
                    h2[0],
                    h2[1],
                    h2[2],
                    h3[0],
                    h3[1],
                    h3[2],
                ]
            )
