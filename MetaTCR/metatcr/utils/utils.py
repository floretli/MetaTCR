import pickle
import os

def save_pk(file_savepath, data):
    with open(file_savepath, "wb") as fp:
        pickle.dump(data, fp)

def load_pkfile(filename):
    with open(filename, "rb") as fp:
        data_dict = pickle.load(fp)
    return data_dict

def read_filelist(filepath_txt):

    if isinstance(filepath_txt, list):
        total_filelist = []
        for f in filepath_txt:
            filelist = open(f, "r").read().split("\n")
            filelist.remove("")
            total_filelist += filelist
    else:
        total_filelist = open(filepath_txt, "r").read().split("\n")
        total_filelist.remove("")

    for filepath in total_filelist:
        if not (os.path.isfile(filepath)):
            total_filelist.remove(filepath)
            print(f"Warning: file {filepath} does not exist. Removing from list.")
    return total_filelist

def class_sampling(group, n, seed=123):
    return group.sample(n=n, random_state=seed)

def class_balance(df, label_col="label"):
    class_size = df[label_col].value_counts().to_list()
    min_size = min(class_size)
    df = df.groupby(label_col).apply(class_sampling, min_size)
    return df

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == "__main__":
    read_filelist("/home/grads/miaozhhuo2/projects/TCRseq_data/datalist/MDA_T.list")
