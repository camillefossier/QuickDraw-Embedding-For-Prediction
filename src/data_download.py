import wget
import os

curr = os.path.dirname(os.path.abspath(__file__))

data_download_path = os.path.join(curr, "../data/")

labels = ["axe", "sword", "squirrel"]

def make_url(label, datatype):
    return "https://storage.cloud.google.com/quickdraw_dataset/full/{}/{}.ndjson".format(datatype, label)

def download(labels, datatype):
    for label in labels:
        quickdraw_url = make_url(label, datatype)
        wget.download(quickdraw_url, data_download_path)    

if __name__ == "__main__":
    if not os.path.isdir(data_download_path):
        os.mkdir(data_download_path)
    download(labels, "raw")