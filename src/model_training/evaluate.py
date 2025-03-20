from torch.utils.data import DataLoader

from src.model.dataset import MaskRCNNDataset
from src.model.model import maskRCNNModel
from src.model.engine import FitterMaskRCNN

def main():
    test_dataset = MaskRCNNDataset("data/test", datatype="eval")
    collate_fn = lambda x: tuple(zip(*x))
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)
    model = maskRCNNModel()
    




if __name__ == '__main__':
    main()