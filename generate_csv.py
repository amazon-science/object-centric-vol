import glob
import csv

glob.glob("./data_ckpt_logs/dataset/ILSVRC2015_224px/train" +"/*"*2)

train_csv = ["/".join(["."] + path.split("/")[-3:]) for path in glob.glob("./data_ckpt_logs/dataset/ILSVRC2015_224px/train" +"/*"*2)]
print("Generate training list csv")
print("Training video number:", len(train_csv))
with open('./data_ckpt_logs/dataset/ILSVRC2015_224px/train.csv', 'w', newline='') as file:
    writer = csv.writer(file, delimiter=' ')
    for item in train_csv:
        writer.writerow([item, 0])

test_csv=["/".join(["."] + path.split("/")[-2:]) for path in glob.glob("./data_ckpt_logs/dataset/ILSVRC2015_224px/test" +"/*"*1)]
print("Generate test list csv")
print("Test video number:", len(test_csv))
with open('./data_ckpt_logs/dataset/ILSVRC2015_224px/test.csv', 'w', newline='') as file:
    writer = csv.writer(file, delimiter=' ')
    for item in test_csv:
        writer.writerow([item, 0])

val_csv=["/".join(["."] + path.split("/")[-2:]) for path in glob.glob("./data_ckpt_logs/dataset/ILSVRC2015_224px/val" +"/*"*1)]
print("Generate val list csv")
print("Validation video number:", len(val_csv))
with open('./data_ckpt_logs/dataset/ILSVRC2015_224px/val.csv', 'w', newline='') as file:
    writer = csv.writer(file, delimiter=' ')
    for item in val_csv:
        writer.writerow([item, 0])