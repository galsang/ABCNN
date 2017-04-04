
import random
with open("msr_paraphrase_train.txt", "r", encoding="utf-8") as f:
    attributes = f.readline()
    sentences = []

    for line in f:
        if int(line.split("\t")[0]) == 1:
            sentences.append(line.split("\t")[3])

    selected = random.sample(sentences, k=100)

    with open("result.txt", "w", encoding="utf-8") as f2:
        print(attributes, file=f2)

        for s in selected:
            print("1\t1\t1\t" + s + "\t" + s, file=f2)
            print("0\t1\t1\t" + s + "\t" + "I have no idea.", file=f2)



