import random
import os
import argparse
import csv
import glob

'''
cspc2018_challenge score
Written by:  Xingyao Wang, Feifei Liu, Chengyu Liu
             School of Instrument Science and Engineering
             Southeast University, China
             chengyu@seu.edu.cn
'''

'''
Save prdiction answers to answers.csv in local path, the first column is recording name and the second
column is prediction label, for example:
Recoding    Result
B0001       1
.           .
.           .
.           .
'''


def cpsc2018(record_base_path):
    # ecg = scipy.io.loadmat(record_path)
    ###########################INFERENCE PART################################

    ## Please process the ecg data, and output the classification result.
    ## result should be an integer number in [1, 9].
    with open('answers.csv', 'w') as csvfile:
        writer = csv.writer(csvfile)
        # column name
        writer.writerow(['Recording', 'Result'])
        for mat_item in os.listdir(record_base_path):
            if mat_item.endswith('.mat') and (not mat_item.startswith('._')):
                result = 5
                # result = random.randint(1, 9)
                ## If the classification result is an invalid number, the result will be determined as normal(1).
                if result > 9 or result < 1 or not (str(result).isdigit()):
                    result = 1
                record_name = mat_item.rstrip('.mat')
                answer = [record_name, result]
                # write result
                writer.writerow(answer)

        csvfile.close()

    ###########################INFERENCE PART################################


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p',
                        '--recording_path',
                        help='path saving test record file')

    args = parser.parse_args()

    result = cpsc2018(record_base_path=args.recording_path)
