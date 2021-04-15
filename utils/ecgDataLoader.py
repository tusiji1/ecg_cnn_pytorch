import torch as th
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import utils.datasolution as ds

project_path = 'G:\\PostGraduate\\pythonProject\\'
data_path = project_path + 'ecg_data\\'


class ecgDataSet(Dataset):
    def __init__(self):
        self.train_x, self.train_y, self.test_x, self.test_y = ds.loadData(data_path)
        self.len = self.train_x.shape[0]

    def __getitem__(self, index):
        return self.train_x[index], self.train_y[index]

    def __len__(self):
        return self.len


dataset = ecgDataSet()


# 数据集 小批量 打乱 线程数
def fun():
    train_loader = DataLoader(dataset=dataset, batch_size=32, shuffle=True, num_workers=1)
    # print(dataset.__getitem__(1),dataset.__len__())
    return train_loader


if __name__ == '__main__':
    fun()
