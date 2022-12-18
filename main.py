import io

import torchdata.datapipes as dp

def read_item(item):
    _, data = item
    return io.BytesIO(data.read())

def create_dataloader(data_location):
    with open(data_location, "r") as f:
        files = f.read().splitlines()
    datapipe = dp.iter.IterableWrapper(files)
    datapipe = datapipe.shuffle()
    datapipe = datapipe.sharding_filter()
    datapipe = dp.iter.S3FileLoader(datapipe)
    datapipe = datapipe.map(read_item)

    return datapipe

def main():
    dataloader = create_dataloader("data.txt")
    for n, x in enumerate(dataloader):
        print(x)
        if n >= 5:
            break

if __name__ == "__main__":
    main()
