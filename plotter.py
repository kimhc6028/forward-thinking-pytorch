import pickle
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')

def subplot(subplot, bp, ft, title, test=True):
    plt.subplot(subplot)

    x = np.arange(0, 40)
    plt.plot(x, bp, color='r', label='backpropagation')
    plt.plot(x, ft, color='g', label='forward-thinking')
    plt.legend(bbox_to_anchor=(1.0, 1.), loc=2, ncol=1, fontsize=15)
    axes = plt.gca()
    axes.set_xlim([0, 40]) if not test else axes.set_xlim([0, 200])
    axes.set_ylim([0, 100])
    plt.title(title, fontsize=20, y = 0.9)
    plt.ylabel('Accuracy',fontsize=15)
    plt.xlabel('Epochs',fontsize=15)
    plt.grid(True)


try:
    f = open('./result/bp.pickle')
    bp_train, bp_test = pickle.load(f)
    f.close()
except:
    print('bp.pickle does not exist! Try \n
    python backpropagation.py')

try:
    f = open('./result/bp_deep.pickle')
    dbp_train, dbp_test = pickle.load(f)
    f.close()
except:
    print('bp_deep.pickle does not exist! Try \n
    python backpropagation.py --deep --epoch 200')

try:
    f = open('./result/ft.pickle')
    ft_train, ft_test = pickle.load(f)
    f.close()
except:
    print('ft.pickle does not exist! Try \n
    python forward_thinking.py')

try:
    f = open('./result/ft_deep.pickle')
    dft_train, dft_test = pickle.load(f)
    f.close()
except:
    print('ft_deep.pickle does not exist! Try \n
    python forward_thinking.py --deep --epoch 200')


subplot('221', bp_train, ft_train, 'Training Accuracy')
subplot('222', bp_test, ft_test, 'Test Accuracy')
subplot('223', dbp_train, dft_train, 'Training Accuracy (deep)')
subplot('224', dbp_test, dft_test, 'Test Accuracy (deep)')

plt.show()
