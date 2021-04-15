import utils.datasolution as dl
import torch.nn as tn
import torch.optim
import ecg_cnn_moudle
import utils.ecgDataLoader as edl
import torch as th

project_path = 'G:\\PostGraduate\\pythonProject\\'
data_path = project_path + 'ecg_data\\'
model = ecg_cnn_moudle.cnn_ecg_model()

def train():
    criterion = tn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
    running_loss = 0.0
    batch_idx = 0
    train_loader=edl.fun()
    for epoch in range(1):
        for i,data in enumerate(train_loader,0):
            inputs,target=data
            #inputs=th.unsqueeze(inputs,1)
            inputs=inputs.float()
            target=target.long()
            inputs=inputs.permute(0,2,1)
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

def test():
    correct=0
    total=0
    test_loader=edl.funTest()
    with th.no_grad():
        for data in test_loader:
            inputs,target=data
            inputs=inputs.float()
            inputs=inputs.permute(0,2,1)
            target=target.long()
            outputs=model(inputs)
            _,predicted=th.max(outputs.data,dim=1)
            total+=target.size(0)
            correct+=(predicted==target).sum().item()
    print(correct)


# 按间距中的绿色按钮以运行脚本。
if __name__ == '__main__':
    train()
    test()
