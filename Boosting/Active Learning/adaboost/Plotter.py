from __future__ import division
import numpy as np
import matplotlib.pyplot as plt


class Plot(object):
    def roc(self, x, y, name):
        auc = -np.trapz(y, x)
        plt.plot(x, y)
        legend = ["{0} AB with AUC = {1:0.3f}".format(name.capitalize(), auc)]
        plt.legend(legend, loc='lower right')
        plt.title("ROC for AdaBoost using Decision stumps")
        plt.ylabel("True Positive Rate")
        plt.xlabel("False Positive Rate")
        plt.grid()
        plt.show()

    def auc(self, auc_round):
        x, y = zip(*auc_round)
        plt.plot(x, y)
        plt.title("AUC for optimal stumps")
        plt.ylabel("AUC")
        plt.xlabel("Round")
        plt.grid()
        plt.show()

    def le(self, auc_round):
        x, y = zip(*auc_round)
        plt.plot(x, y)
        plt.title("Round Error")
        plt.ylabel("Error")
        plt.xlabel("Round")
        plt.grid()
        plt.show()

    def error(self, tri_tuple):
        x, y1, y2 = zip(*tri_tuple)
        legend = []

        plt.plot(x, y1)
        legend.append("Testing Error")

        plt.plot(x, y2)
        legend.append("Training Error")

        plt.legend(legend, loc="upper right")
        plt.title("Training/Testing Error per Round")
        plt.ylabel("Error")
        plt.xlabel("Round")
        plt.grid()
        plt.show()