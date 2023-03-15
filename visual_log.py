import matplotlib.pyplot as plt
import numpy as np
import math

def show_joint_better():
    fig = plt.figure()

    read = open('/apdcephfs/private_v_mbxue/dual/awesome-dual-learning/output/general/joint/2021-06-02-16-00/log/exp.log', 'r', encoding='utf-8-sig').readlines()
    joint_R = []
    joint_G = []
    for line in read:
        if 'Loss' not in line:
            continue
        idx1 = line.find('R:') + 2
        idx2 = idx1 + line[idx1:].find(' (')
        joint_R.append(math.log(float(line[idx1:idx2])))
        idx1 = line.find('G:') + 2
        idx2 = idx1 + line[idx1:].find(' (')
        joint_G.append(math.log(float(line[idx1:idx2])))

    read = open('/apdcephfs/private_v_mbxue/dual/awesome-dual-learning/output/general/recognizer/2021-06-01-17-54/log/exp.log', 'r', encoding='utf-8-sig').readlines()
    R = []
    for line in read:
        if 'Loss' not in line:
            continue
        idx1 = line.find('Loss ') + 5
        idx2 = idx1 + line[idx1:].find(' (')
        R.append(math.log(float(line[idx1:idx2])))

    read = open('/apdcephfs/private_v_mbxue/dual/awesome-dual-learning/output/general/generator/2021-06-02-19-42/log/exp.log', 'r', encoding='utf-8-sig').readlines()
    G = []
    for line in read:
        if 'G:' not in line:
            continue
        idx1 = line.find('G:') + 2
        idx2 = idx1 + line[idx1:].find(' (')
        G.append(math.log(float(line[idx1:idx2])))
    print(G[:10])
    ax1 = fig.add_subplot(211)
    idx = np.array([i for i in range(len(G))])
    ax1.plot(idx, G, label = 'G')
    ax1.plot(idx, joint_G[:len(G)], label = 'joint G')
    ax1.xlabel('step 8/epoch')

    ax2 = fig.add_subplot(212)
    idx = np.array([i for i in range(len(R))])
    ax2.plot(idx, R, label ='R')
    idx = np.array([i for i in range(len(R))])
    ax2.plot(idx, joint_R[:len(R)], label = 'joint R')

    plt.savefig('log.png')

def compare_joints():
    fig = plt.figure(figsize=(12,9))

    def get_data(logpath, name):
        joint_R = []
        ACC = []
        joint_G = []
        Loss = []
        if isinstance(logpath, str):
            read = open(logpath, 'r', encoding='utf-8-sig').readlines()
        elif isinstance(logpath, list): 
            read = []
            for f in logpath:
                r = open(f, 'r', encoding='utf-8-sig').readlines()
                read = read + r
        else:
            raise RuntimeError
        for line in read:
            if 'Loss' in line:
                idx1 = line.find('R:') + 2
                idx2 = idx1 + line[idx1:].find(' (')
                joint_R.append(math.log(float(line[idx1:idx2])))
                idx1 = line.find('G:') + 2
                idx2 = idx1 + line[idx1:].find(' (')
                joint_G.append(math.log(float(line[idx1:idx2])))
            elif 'Test accuray: ' in line:
                idx1 = line.find('Test accuray: ') + len('Test accuray: ')
                idx2 = line.find('\n')
                ACC.append(float(line[idx1:idx2])*100)
            elif 'Test loss: ' in line:
                idx1 = line.find('Test loss: ') + len('Test loss: ')
                idx2 = line.find('\n')
                Loss.append(math.log(float(line[idx1:idx2])))
        return {'R':joint_R[::5], 'G':joint_G, 'ACC':ACC, 'Loss':Loss, 'label':name}

    def make_figure(dicts):
        ax1 = fig.add_subplot(211)
        for d in dicts:
            idx = np.array([i for i in range(len(d['G']))])
            ax1.plot(idx, d['G'], label=d['label'] + ' train', linewidth=1.0)
        for d in dicts:
            num = min(d['G'])
            min_train = (d['G'].index(num), num)
            times = len(d['G'])//len(d['Loss'])
            print('loss',len(d['Loss']),len(d['G']),  len(d['Loss'])*times)
            idx = np.array([i for i in range(0, len(d['Loss'])*times, times)])
            num = min(d['Loss'])
            min_test = (d['Loss'].index(num), num)
            ax1.plot(idx, d['Loss'], label=d['label'] + ' test', linewidth=2, linestyle='--', markevery=[min_test[0]], marker='o', markersize=10)
            
        ax1.set_xlabel('Step (N/epoch)')
        ax1.set_ylabel('Loss (ln)')
        # ax1.annotate(text='{:.3f}'.format(min_test[1] ), xy=min_test, xytext=(30, 30),weight='bold', arrowprops=dict(arrowstyle='-|>',connectionstyle='arc3'), textcoords='offset points')
        ax1.legend()

        ax2 = fig.add_subplot(212)
        ax2b = ax2.twinx()
        for d in dicts:
            idx = np.array([i for i in range(len(d['R']))])
            ax2.plot(idx, d['R'], label=d['label'] + 'Loss', linewidth=1.0)
            times = len(d['R'])//len(d['ACC'])
            print('ACC', len(d['ACC']),len(d['R']),  len(d['ACC'])*times)
            idx = np.array([i for i in range(0, times*len(d['ACC']), times)])
            num = max(d['ACC'])
            max_acc = (d['ACC'].index(num), num)
            ax2b.plot(idx, d['ACC'], label=d['label'] + ' ACC', linewidth=1.5, linestyle='--', markevery=[max_acc[0]], marker='o',  markersize=10)
            
        ax2.set_xlabel('Step (N/epoch)')
        ax2.set_ylabel('Loss (ln)')
        ax2b.set_ylabel('ACC (%)')
        # ax2b.annotate(text='{:.3f}'.format(max_acc[1]), xy=max_acc, xytext=(-30, -50),weight='bold', arrowprops=dict(arrowstyle='-|>',connectionstyle='arc3'), textcoords='offset points')
        ax2.legend()
        ax2b.legend()

    dicts = []
    dicts.append(get_data('/apdcephfs/private_v_mbxue/dual/awesome-dual-learning/output/general/joint/2021-06-30-16-01/log/exp.log', 'without TPE'))
    dicts.append(get_data('/apdcephfs/private_v_mbxue/dual/awesome-dual-learning/output/general/joint/2021-06-16-19-15/log/exp.log', 'with TPE'))
    make_figure(dicts)
    plt.savefig('log.png')

compare_joints()

