import os
import numpy as np
import matplotlib.pyplot as plt

def draw_mcep(mcep_file1, mcep_file2, order):
    mcep1 = np.loadtxt(mcep_file1)
    mcep2 = np.loadtxt(mcep_file2)
    plt.plot(mcep1[:, order], label=mcep_file1)
    plt.plot(mcep2[:, order], label=mcep_file2)
    plt.xlabel("frame")
    plt.title("%dth-order mcep" % order)
    plt.legend(loc=4)
    plt.show()

mcep_file1 = "mcep_my_ali/voice.mcep"
mcep_file2 = "mcep_miku_ali/voice.mcep"
draw_mcep(mcep_file1, mcep_file2, 0)