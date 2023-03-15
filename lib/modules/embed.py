import torch
import numpy as np
import math


def get_sinusoid_sequence_table(n_position, num_hidden, dim=1, xy=True):
    def cal_angle(position, hid_idx):
        return position / np.power(10000, 2 * hid_idx / num_hidden)

    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(num_hidden)]

    sinusoid_table = np.array([get_posi_angle_vec(pos_i)
                               for pos_i in range(n_position)])
    if dim == 1:
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])
    elif dim ==2:
        if xy:
            sinusoid_table[:, :] = np.sin(sinusoid_table[:, :])
        else:
            sinusoid_table[:, :] = np.cos(sinusoid_table[:, :])
    return torch.FloatTensor(sinusoid_table)

# tree
def get_sinusoid_tree_table(NUM_DEPTH, num_hidden):
    def cal_angle(position, hid_idx):
        return math.sin(
            (position + 1) * 3.1415926 * 2 * hid_idx / num_hidden
        )
    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(num_hidden)]

    n_position = 2**NUM_DEPTH - 1
    table = np.array([get_posi_angle_vec(pos_i) for pos_i in range(n_position)])

    sinusoid_table = np.zeros((2**NUM_DEPTH-1, num_hidden), dtype=np.float32)
    
    # return torch.FloatTensor(sinusoid_table)

    def Traversal(node, depth):
        back = [node]
        if depth == NUM_DEPTH:
            return [node]
        b0 = Traversal(node+[0], depth+1)
        b1 = Traversal(node+[1], depth+1)
        return back + b0 + b1

    def node2num(List):
        result = []
        for node in List:
            root = 0
            if node == [0]:
                pass
            else:
                for n in node[1:]:
                    root = root * 2 + 1 + n
            result.append(root)
        return result
    
    def node2idx(List):
        result = []
        for node in List:
            root = []
            for idx,n in enumerate(node):
                if idx == 0:
                    root.append(0)
                else:
                    n = 0 if n%2==1 else 1
                    root.append(idx*2 - n)
            result.append(root)
        return result
    
    L = Traversal([0],1)
    R = node2num(L)
    I = node2idx(L)

    for i,r in zip(I,R):
        emb = table[i,:]
        emb = np.mean(emb, axis=0)
        sinusoid_table[r,:] = emb

    return torch.FloatTensor(sinusoid_table)

def visual():
    import matplotlib.pyplot as plt
    def draw(fig, y, name, idx):
        x = np.linspace(1, y.shape[1], y.shape[1])
        ax = fig.add_subplot(2,2,idx)
        ax.set(title=name)
        for i in range(min(3,y.shape[0])):
            ax.plot(x,y[i,:])

    pos_1d = get_sinusoid_sequence_table(16, 128, dim=1).numpy()
    pos_2d_x = get_sinusoid_sequence_table(4, 128, dim=2, xy=True).numpy()
    pos_2d_y = get_sinusoid_sequence_table(4, 128, dim=2, xy=False).numpy()
    pos_tree = get_sinusoid_tree_table(3, 128).numpy()
    fig = plt.figure()

    draw(fig, pos_1d, '1d', 1)
    draw(fig, pos_2d_x, '2d x', 2)
    draw(fig, pos_2d_y, '1d y', 3)
    draw(fig, pos_tree, 'tree', 4)
    plt.savefig('a.png')
