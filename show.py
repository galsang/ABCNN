
import matplotlib.pyplot as plt
import numpy as np

with open("./experiments/WikiQA-ABCNN1-2-LR.txt", "r", encoding="utf-8") as f:
    f.readline()
    MAPs, MRRs = [], []

    for line in f:
        MAP = line[:-1].split("\t")[1]
        MRR = line[:-1].split("\t")[2]

        MAPs.append(MAP)
        MRRs.append(MRR)

print("max:", max(MAPs), max(MRRs))

plt.plot(np.arange(1, len(MAPs)+1, 1), MAPs, 'r')
plt.plot(np.arange(1, len(MAPs)+1, 1), MRRs, 'b')
plt.legend(["MAP", "MRR"])
plt.show()

