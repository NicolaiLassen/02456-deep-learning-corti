import pickle
import matplotlib.pyplot as plt

if __name__ == '__main__':
    pickle_in = open("../ckpt_acc/losses/epoch_batch_losses_e_4000_b.pkl", "rb")
    example_dict = pickle.load(pickle_in)
    plt.plot(example_dict)
    plt.show()