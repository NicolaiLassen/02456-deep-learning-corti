import os
import pickle

import matplotlib.pyplot as plt

if __name__ == "__main__":

    PATH_con = "ckpt_con/losses_epoch"
    PATH_tri = "ckpt_triplet/losses_epoch"
    PATH_contri = "ckpt_con_triplet/losses_epoch"
    PATHS = [PATH_con, PATH_tri, PATH_contri]

    loss_con = []
    loss_tri = []
    loss_contri = []
    for path in PATHS:
        epochs = len(os.listdir(path))

        tmp_PATH = path + "/" + "epoch_mean_losses_e_{}".format(epochs - 1) + ".pkl"
        print(tmp_PATH)
        with (open(tmp_PATH, "rb")) as openfile:
            loss = pickle.load(openfile)
        if path == PATH_con:
            loss_con.append(loss)
        elif path == PATH_tri:
            loss_tri.append(loss)
        elif path == PATH_contri:
            loss_contri.append(loss)

    plt.plot(loss_contri[0])
    plt.title("Contrastive + Triplet loss")
    plt.xlabel("Epoch")
    plt.show()

    # with (open("epoch_mean_losses_e_37.pkl", "rb")) as openfile:
    #     loss = pickle.load(openfile)
    #
    # plt.plot(loss)
    # plt.show()
