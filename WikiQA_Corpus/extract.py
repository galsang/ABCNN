
with open("WikiQA-train.txt", "r", encoding="utf-8") as f:
    QA_pairs = {}

    for line in f:
        s1 = line.split("\t")[0]
        s2 = line.split("\t")[1]
        label = int(line.split("\t")[2])

        if s1 in QA_pairs.keys():
            QA_pairs[s1].append((s2,label))
        else:
            QA_pairs[s1] = [(s2,label)]

    with open("WikiQA-train_real.txt", "w", encoding="utf-8") as f2:
        c = 0

        for s1 in QA_pairs:
            has_answers = 0

            for s2, label in QA_pairs[s1]:
                if label == 1 :
                    c += len(QA_pairs[s1])
                    has_answers = 1
                    break

            if has_answers:
                for s2, label in QA_pairs[s1]:
                    print(s1 + "\t" + s2 + "\t" + str(label), file=f2)

    print("count:", c)