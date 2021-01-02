### NOW COMPILING ALL THE RESULTS TOGETHER
import os
import csv
import re

base_dir = "/share/PI/rbaltman/dnsosa/projects/drug-lit-contradictory-claims/output/trained_biobert"
all_res_file = os.path.join(base_dir, "all_results.csv")

acc = None
precs = None
recs = None
f1s = None

with open(all_res_file, 'w') as out_file:
    res_nums = []
    for res_dir in os.listdir(base_dir):
        if not os.path.isdir(os.path.join(base_dir, res_dir)):
            continue
        else:
            hyp_params = [res_dir]
            hyp_params.extend(re.findall(r"[-+]?\d*\.\d+|\d+", ' '.join(res_dir.split('_')[-4::])))
        with open(os.path.join(os.path.join(base_dir, res_dir), "summary_report.txt"), 'r') as res_file:
            line_nums = []
            for i, line in enumerate(res_file):
                if i == 11:
                    str_num = line.split(' ')[1].split('\n')[0]
                    line_nums.extend([float(str_num)])
                elif 12 <= i <= 14:
                    str_nums = ' '.join(line.split('[')[1].split(']')[0].split()).split()
                    nums = [float(num) for num in str_nums]
                    line_nums.extend(nums)
        hyp_params.extend(line_nums)
        res_nums.append(hyp_params)
        
    write = csv.writer(out_file)     
    write.writerows(res_nums)            

