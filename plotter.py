import pickle
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')

def subplot(subplot, bp, ft, title, test=True):
    plt.subplot(subplot)
    '''
    """This averages training result for every [norm_range] results"""
    if not test:
        norm_range = 10
        bp = [np.mean(bp[i:i+norm_range]) for i in range(len(bp)/norm_range)]
        ft = [np.mean(ft[i:i+norm_range] )for i in range(len(ft)/norm_range)]
    '''
    if not (len(bp) == len(ft)):
        print('Warning : Length of bp and ft are not equal. This may cause distortion in graph')
    x = np.arange(0, len(bp))
    plt.plot(x, bp, color='r', label='backpropagation')
    plt.plot(x, ft, color='g', label='forward-thinking')
    plt.legend(bbox_to_anchor=(1.0, 1.), loc=2, ncol=1, fontsize=20)
    axes = plt.gca()
    axes.set_ylim([0, 100])
    plt.title(title, fontsize=40, y = 1.05)
    plt.ylabel('Accuracy', fontsize=30)
    plt.xlabel('Epochs' if test else 'minibatches', fontsize=30)
    plt.grid(True)


try:
    f = open('./result/bp.pickle')
    bp_train, bp_test = pickle.load(f)
    f.close()
except:
    print('bp.pickle does not exist! Try \n python backpropagation.py')

try:
    f = open('./result/bp_deep.pickle')
    dbp_train, dbp_test = pickle.load(f)
    f.close()
except:
    print('bp_deep.pickle does not exist! Try \n python backpropagation.py --deep --epoch 200')

try:
    f = open('./result/ft.pickle')
    ft_train, ft_test = pickle.load(f)
    f.close()
except:
    print('ft.pickle does not exist! Try \n python forward_thinking.py')

try:
    f = open('./result/ft_deep.pickle')
    dft_train, dft_test = pickle.load(f)
    f.close()
except:
    print('ft_deep.pickle does not exist! Try \n python forward_thinking.py --deep --epoch 200')


subplot('221', bp_train, ft_train, 'Training Accuracy', False)
subplot('222', bp_test, ft_test, 'Test Accuracy', True)
subplot('223', dbp_train, dft_train, 'Training Accuracy (deep)', False)
subplot('224', dbp_test, dft_test, 'Test Accuracy (deep)', True)

plt.show()
