import matplotlib.pyplot as plt, matplotlib.cm as cm

class window:
    def __init__(self):
        plt.ion()
        plt.show()

    def display_prepost(self, pre_im, pos_im, display_difference = True, save_to = None):
        nimages = 3 if display_difference else 2
        plt.subplot(1, nimages, 1)
        plt.title("Prefocal")
        plt.imshow(pre_im, cmap=cm.gray)
        plt.subplot(1, nimages, 2)
        plt.title("Postfocal")
        plt.imshow(pos_im, cmap=cm.gray)
        if display_difference:
            plt.subplot(1, nimages, 3)
            plt.title("Difference")
            plt.imshow(pre_im - pos_im, cmap=cm.gray)
        plt.draw()
        plt.pause(0.001)

        if save_to is not None:
            plt.savefig(save_to)

def display_prepost(pre_im, pos_im, display_difference = True, save_to = None):
    nimages = 3 if display_difference else 2
    plt.subplot(1, nimages, 1)
    plt.title("Prefocal")
    plt.imshow(pre_im, cmap=cm.gray)
    plt.subplot(1, nimages, 2)
    plt.title("Postfocal")
    plt.imshow(pos_im, cmap=cm.gray)
    if display_difference:
        plt.subplot(1, nimages, 3)
        plt.title("Difference")
        plt.imshow(pre_im - pos_im, cmap=cm.gray)
    plt.show()

    if save_to is not None:
        plt.savefig(save_to)