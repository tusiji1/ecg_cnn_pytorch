import utils.datasolution as dl
import torch.nn as tn
import torch.optim
import ecg_cnn_moudle

project_path = 'G:\\PostGraduate\\pythonProject\\'
data_path = project_path + 'ecg_data\\'
model = ecg_cnn_moudle.cnn_ecg_model()


def getData():
    return dl.loadData(data_path)


def train():
    criterion = tn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
    inputs, target, test_inputs, test_target = getData()
    train_data = [inputs, target]
    running_loss = 0.0
    batch_idx = 0
    for i in range(len(train_data)):
        optimizer.zero_grad()
        inputs = train_data[0][i]
        target = train_data[1][i]
        outputs = model(inputs)
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        batch_idx += 1
        if batch_idx % 300 == 299:
            print('%5d loss:%.3f' % (batch_idx + 1, running_loss / 2000))
            running_loss = 0.0


# 按间距中的绿色按钮以运行脚本。
if __name__ == '__main__':
    train()
