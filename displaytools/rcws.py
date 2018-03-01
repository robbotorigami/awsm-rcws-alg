import matplotlib.pyplot as plt, matplotlib.cm as cm

class window:
    def __init__(self, updatable = True):
        self.updatable = updatable
        if updatable:
            plt.ion()
            plt.show()

    def display_prepost(self, pre_im, pos_im, display_difference = True, save_to = None, cross_section = False):
        nimages = 3 if display_difference else 2
        nrows = 2 if cross_section else 1
        plt.subplot(nrows, nimages, 1)
        plt.title("Prefocal")
        plt.imshow(pre_im, cmap=cm.gray)
        plt.subplot(nrows, nimages, 2)
        plt.title("Postfocal")
        plt.imshow(pos_im, cmap=cm.gray)
        if display_difference:
            plt.subplot(nrows, nimages, 3)
            plt.title("Difference")
            plt.imshow(pre_im - pos_im, cmap=cm.gray)

        if cross_section:
            plt.subplot(nrows, nimages, 4)#(nrows-1)*nimages + 1)
            plt.cla()
            plt.ylim((-1,1))
            plt.gca().axes.get_xaxis().set_visible(False)
            plt.gca().axes.get_yaxis().set_visible(False)
            plt.plot([x for x in range(pre_im.shape[1])], pre_im[pre_im.shape[0]//2], 'r')
            plt.subplot(nrows, nimages, 5)#(nrows-1)*nimages + 2)
            plt.cla()
            plt.ylim((-1,1))
            plt.gca().axes.get_xaxis().set_visible(False)
            plt.gca().axes.get_yaxis().set_visible(False)
            plt.plot([x for x in range(pos_im.shape[1])], pos_im[pos_im.shape[0]//2], 'r')

            if display_difference:
                diff = pre_im - pos_im
                plt.subplot(nrows, nimages, 6)#(nrows-1)*nimages + 3)
                plt.cla()
                plt.ylim((-1,1))
                plt.gca().axes.get_xaxis().set_visible(False)
                plt.gca().axes.get_yaxis().set_visible(False)
                plt.plot([x for x in range(diff.shape[1])], diff[diff.shape[0] // 2], 'r')
        if self.updatable:
            plt.draw()
            plt.pause(0.001)
        else:
            plt.show()

        if save_to is not None:
            plt.savefig(save_to)

    def display_prepost_masks(self, pre_im, pos_im, masks):
        nimages = 3
        nrows = 2
        plt.subplot(nrows, nimages, 1)
        plt.title("Prefocal")
        plt.imshow(pre_im, cmap=cm.gray)
        plt.subplot(nrows, nimages, 2)
        plt.title("Postfocal")
        plt.imshow(pos_im, cmap=cm.gray)
        plt.subplot(nrows, nimages, 3)
        plt.title("Difference")
        plt.imshow(pre_im - pos_im, cmap=cm.gray)
        plt.subplot(nrows, nimages, 4)
        plt.title("Prefocal Mask")
        plt.imshow(masks[0], cmap=cm.gray)
        plt.subplot(nrows, nimages, 5)
        plt.title("Postfocal Mask")
        plt.imshow(masks[1], cmap=cm.gray)
        plt.subplot(nrows, nimages, 6)
        plt.title("Combined Mask")
        plt.imshow(masks[2], cmap=cm.gray)
        if self.updatable:
            plt.draw()
            plt.pause(0.001)
        else:
            plt.show()

    def display_features(self, pre_im, pos_im, laplacian, normals):
        nimages = 3
        nrows = 2
        plt.subplot(nrows, nimages, 1)
        plt.title("Prefocal")
        plt.imshow(pre_im, cmap=cm.gray)
        plt.subplot(nrows, nimages, 2)
        plt.title("Postfocal")
        plt.imshow(pos_im, cmap=cm.gray)
        plt.subplot(nrows, nimages, 3)
        plt.title("Difference")
        plt.imshow(pre_im - pos_im, cmap=cm.gray)
        plt.subplot(nrows, nimages, 4)
        plt.title("Laplacian")
        plt.imshow(laplacian, cmap=cm.gray)
        plt.subplot(nrows, nimages, 5)
        plt.title("Normals")
        plt.imshow(normals, cmap=cm.gray)
        plt.subplot(nrows, nimages, 6)
        plt.title("Combined")
        plt.imshow(normals + laplacian, cmap=cm.gray)
        if self.updatable:
            plt.draw()
            plt.pause(0.001)
        else:
            plt.show()

    def display_centroid(self, pre_im, pos_im, centroids):
        plt.subplot(1, 2, 1)
        plt.title("Prefocal")
        plt.imshow(pre_im, cmap=cm.gray)
        if centroids is not None:
            plt.plot(centroids[0][0], centroids[0][1], 'ro')
        plt.subplot(1, 2, 2)
        plt.title("Postfocal")
        plt.imshow(pos_im, cmap=cm.gray)
        if centroids is not None:
            plt.plot(centroids[1][0], centroids[1][1], 'ro')

        if self.updatable:
            plt.draw()
            plt.pause(0.001)
        else:
            plt.show()