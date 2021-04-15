import utils.datasolution as dl
import torch.nn as tn
import torch.optim
import ecg_cnn_moudle
import utils.ecgDataLoader as edl
import torch as th

project_path = 'G:\\PostGraduate\\pythonProject\\'
data_path = project_path + 'ecg_data\\'
model = ecg_cnn_moudle.cnn_ecg_model()
th.tensor

def getData():
    return dl.loadData(data_path)


def train():
    criterion = tn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
    running_loss = 0.0
    batch_idx = 0
    train_loader=edl.fun()
    for epoch in range(10):
        for i,data in enumerate(train_loader,0):
            inputs,target=data
            inputs=th.unsqueeze(inputs,1)
            inputs=inputs.float()
            outputs = model(inputs)
            loss = criterion(outputs, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # running_loss += loss.item()
        # batch_idx += 1
        # if batch_idx % 300 == 299:
        #     print('%5d loss:%.3f' % (batch_idx + 1, running_loss / 2000))
        #     running_loss = 0.0


# 按间距中的绿色按钮以运行脚本。
if __name__ == '__main__':
    train()
