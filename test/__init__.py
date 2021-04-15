import torch as th

conv1 = th.nn.Conv1d(in_channels=256,out_channels=100,kernel_size=2)
input = th.randn(32,35,256)
# batch_size x text_len x embedding_size -> batch_size x embedding_size x text_len
input = input.permute(0,2,1)
out = conv1(input)
print(out.size())
