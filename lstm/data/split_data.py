import os
DATA_DIR = 'foilv1.0'
#DATA_DIR = 'newdata'
TRAIN_FILE = 'foil_training_r8_style.txt'
#TRAIN_FILE = 'mc_mscoco_train_foilformat_sentencelabel.txt'
TEST_FILE = 'foil_testing_r8_style.txt'
#TEST_FILE = 'mc_mscoco_test_foilformat_sentencelabel.txt'
#TEST_FILE = 'foil_training_r8_style.txt'
TRAID_DIR = 'train_txt_foil'
TEST_DIR = 'test_txt_foil'

if __name__=='__main__':
    train_file = []
    fp = open(os.path.join(DATA_DIR, TRAIN_FILE), 'r')
    labels = {}
    count = 0
    train_label = []
    train_file = []
    for lines in fp:
        label = lines.strip().split('\t')[0]
        txt = lines.replace(label, '')
        if label not in labels:
            labels[label] = len(labels)
        count += 1
        # writing '#count.txt' file
        filename = str(count)+'.txt'
        fp_train = open(os.path.join(TRAID_DIR, filename), 'wb')
        train_file.append(filename)
        #print txt
        fp_train.write(txt)
        fp_train.close()
        # record #count label
        train_label.append(labels[label])
    fp_file = open('train_txt_foil.txt', 'w')
    for file in train_file:
       # print file
        fp_file.write(file + '\n')
    fp_file.close()
    fp_label = open('train_label_foil.txt', 'w')
    for t in train_label:
       # print t
        fp_label.write(str(t) + '\n')
    fp_label.close()

    fp.close()
    print(labels)
    fp = open(os.path.join(DATA_DIR, TEST_FILE), 'r')
    count = 0
    test_label = []
    test_file = []
    for lines in fp:

        label = lines.strip().split('\t')[0]
        #label = lines.split()[0].strip()
        txt = lines.replace(label, '')
        count += 1
        # writing '#count.txt' file
        filename = str(count)+'.txt'
        fp_test = open(os.path.join(TEST_DIR, filename), 'wb')
        test_file.append(filename)
        fp_test.write(txt)
        fp_test.close()
        # record #count label
        test_label.append(labels[label])
    fp_file = open('test_txt_foil.txt', 'w')
    for file in test_file:
        fp_file.write(file + '\n')
    fp_file.close()

    fp_label = open('test_label_foil.txt', 'w')
    for t in test_label:
        fp_label.write(str(t) + '\n')
    fp_label.close()

    fp.close()

