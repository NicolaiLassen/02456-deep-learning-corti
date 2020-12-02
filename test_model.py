import pickle
import matplotlib.pyplot as plt

with open('./ckpt/losses_batch/epoch_batch_losses_e_9_b_38.pkl', 'rb') as f:
    x = pickle.load(f)
    plt.plot(x)
    print(x)
    plt.show()
