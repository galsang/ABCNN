import numpy as np

with open("msr_paraphrase_train_original.txt", "r", encoding="utf-8") as f:
    attributes = f.readline()
    lines = f.readlines()
    np.random.shuffle(lines)

    with open("msr_paraphrase_dev.txt", "w", encoding="utf-8") as f2:
        print(attributes, file=f2, end="")
        for l in lines[:400]:
            print(l, file=f2, end="")

    with open("msr_paraphrase_train.txt","w", encoding="utf-8") as f3:
        print(attributes, file=f3, end="")
        for l in lines[400:]:
            print(l, file=f3, end="")
