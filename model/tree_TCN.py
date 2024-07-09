import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
from treelib import Tree, Node

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        """
        其实这就是一个裁剪的模块，裁剪多出来的padding
        """
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2,shrink_flag=False,ratio =4):
        """
        相当于一个Residual block

        :param n_inputs: int, 输入通道数
        :param n_outputs: int, 输出通道数
        :param kernel_size: int, 卷积核尺寸
        :param stride: int, 步长，一般为1
        :param dilation: int, 膨胀系数
        :param padding: int, 填充系数
        :param dropout: float, dropout比率
        """
        super(TemporalBlock, self).__init__()
        # self.shrink_flag = shrink_flag

        # add BN instead WN, when using network slimming, use self.net instead of self.net2
        # self.conv2 = nn.Conv1d(n_inputs, n_outputs, kernel_size,stride=stride, padding=padding, dilation=dilation)
        # self.bn = nn.BatchNorm1d(n_outputs)
        # nn.init.constant_(self.bn.weight, 0.5)



        # 经过conv1，输出的size其实是(Batch, input_channel, seq_len + padding)
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)  # 裁剪掉多出来的padding部分，维持输出时间步为seq_len
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)  # 裁剪掉多出来的padding部分，维持输出时间步为seq_len
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)



        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        # self.net = nn.Sequential(self.conv1, self.bn, self.chomp1, self.relu1, self.dropout1)
        self.net2 = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1)
        self.relu = nn.ReLU()


        self.relu3 = nn.ReLU()
        self.squeeze = nn.AdaptiveAvgPool1d(1)
        self.compress = nn.Conv1d(n_inputs, n_inputs // ratio, 1, 1, 0)
        self.excitation = nn.Conv1d(n_inputs // ratio, n_inputs, 1, 1, 0)
        self.sig1 = nn.Sigmoid()
        self.se_scale = nn.Sequential(self.squeeze,self.compress,self.relu3,self.excitation,self.sig1)
        self.init_weights()

    def init_weights(self):
        """
        参数初始化

        :return:
        """
        self.conv1.weight.data.normal_(0, 0.01)
        # self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        """
        :param x: size of (Batch, input_channel, seq_len)
        :return:x
        """
        # if self.shrink_flag==False:
        #     scale = self.se_scale(x)
        #     scale_x = torch.mul(x,scale)
        #     out = self.net(scale_x)
        # else:

        out = self.net2(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


def TreetoList(tree):
    all_nodes = tree.all_nodes()
    data_list = []
    for node in all_nodes:
        if node.identifier!='root':
            data= node.data[:]
            data.append(tree.parent(node.identifier).data[-1])
            data_list.append(data)
    return  data_list




class TemporalConvNet_v1(nn.Module):
    def __init__(self,num_inputs,num_outputs,tree,dropout=0.2):
        super(TemporalConvNet_v1, self).__init__()

        data_list_v1 = TreetoList(tree)
        self.linear1 = nn.Linear(num_inputs,num_outputs)

        #data_list[i]{0:kernel_size,1:flag,2:dilation_bottom,3:level,4:out_channel 5:in_channel}
        self.tree_nodes = nn.ModuleList(
            [TemporalBlock(num_inputs if level==0 else in_channel, out_channel,ks,stride=1,
                           dilation=db**level,padding=(ks - 1) * db**level,dropout=dropout)
             for (ks,flag,db,level,out_channel,in_channel) in data_list_v1])
        self.tree_Linear = nn.ModuleList([nn.Linear(out_channel,num_outputs) for (_,_,_,_,out_channel,_) in data_list_v1])
        self.linear2 = nn.ModuleList([nn.Linear(i,1) for i in range(1,len(data_list_v1)+2)]) # 用于剪枝的线性层
        self.linear3 = nn.Linear(12,1) # 用于测试

        for linear in self.linear2:
            nn.init.kaiming_normal_(linear.weight)


    def forward(self,input,dic_child,dic_father,tree_len,mask_ids=[]):
        dic_data ={}
        queue = ['root']
        input = input.transpose(1,2)
        while queue:
            node_id = queue.pop(0)
            if node_id =='root':
                dic_data['root'] = input
            else:
                terminal_input = dic_father[node_id] #上一层输出的key
                terminal_data = self.tree_nodes[node_id](dic_data[terminal_input])
                dic_data[node_id]  = terminal_data
            if node_id in dic_child:
                for child in dic_child[node_id]:
                    queue.append(child)

        insert_flag = 0
        for key,value in dic_data.items():
            if key =='root':
                if key not in mask_ids:
                    final = self.linear1(value.transpose(1, 2)).unsqueeze(-1)
                    insert_flag+=1

            elif key not in mask_ids:
                if insert_flag == 0:
                    final = self.tree_Linear[key](value.transpose(1,2)).unsqueeze(-1)
                    insert_flag += 1
                else:
                    linear_transfer = self.tree_Linear[key](value.transpose(1, 2)).unsqueeze(-1)
                    final = torch.cat((final, linear_transfer), dim=-1)


        middle_len = tree_len-len(mask_ids)
        final_squeeze = self.linear2[middle_len-1](final).squeeze(-1)
        return final_squeeze[:,-1,:].contiguous()






class TemporalConvNet_v2(nn.Module):
    def __init__(self,num_inputs,num_channels,tree,dropout=0.2):
        super(TemporalConvNet_v2, self).__init__()
        all_nodes = tree.all_nodes()
        data_list = []
        for node in all_nodes:
            data_list.append(node.data)
        terminal_dim = len(data_list)
        data_list_v1 = data_list[1:]


        self.linear1 = nn.Linear(num_inputs,3)
        self.tree_nodes = nn.ModuleList(
            [TemporalBlock(num_inputs if level==0 else num_channels[level - 1], num_channels[level],kernel_size,stride=1,
                           dilation=dilation_bottom**level,padding=(kernel_size - 1) * dilation_bottom**level,dropout=dropout)
             for (kernel_size,flag,dilation_bottom,level,_) in data_list_v1])
        self.tree_Linear = nn.ModuleList([nn.Linear(num_channels[level],3) for (_,_,_,level,_) in data_list_v1])
        self.linear2 = nn.Linear(terminal_dim,1)
        # self.linear3 = nn.Linear(terminal_dim,1)
        # nn.init.kaiming_normal_(self.linear2.weight)


    def forward(self,input,dic_child,dic_father):#(node_id和treenode[]一一对应)
        dic_data ={}
        queue = ['root']
        input = input.transpose(1,2)
        while queue:
            node_id = queue.pop(0)
            if node_id =='root':
                dic_data['root'] = input
            else:
                terminal_input = dic_father[node_id] #上一层输出的key
                terminal_data = self.tree_nodes[node_id](dic_data[terminal_input])
                dic_data[node_id]  = terminal_data
            if node_id in dic_child:
                for child in dic_child[node_id]:
                    queue.append(child)

        for key,value in dic_data.items():
            if key =='root':
                final = self.linear1(value.transpose(1, 2)).unsqueeze(-1)
            else:
                linear_transfer = self.tree_Linear[key](value.transpose(1, 2)).unsqueeze(-1)
                final = torch.cat((final, linear_transfer), dim=-1)
        final_squeeze = self.linear2(final).squeeze()

        return final_squeeze[:,-1,:].contiguous()


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2,shrink_flag=False):
        """
        TCN，目前paper给出的TCN结构很好的支持每个时刻为一个数的情况，即sequence结构，
        对于每个时刻为一个向量这种一维结构，勉强可以把向量拆成若干该时刻的输入通道，
        对于每个时刻为一个矩阵或更高维图像的情况，就不太好办。

        :param num_inputs: int， 输入通道数
        :param num_channels: list，
        :param kernel_size: int, 卷积核尺寸
        :param dropout: float, drop_out比率
        """
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i  # 膨胀系数：1，2，4，8……
            in_channels = num_inputs if i == 0 else num_channels[i - 1]  # 确定每一层的输入通道数
            out_channels = num_channels[i]  # 确定每一层的输出通道数
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size - 1) * dilation_size, dropout=dropout,shrink_flag=shrink_flag)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        """
        输入x的结构不同于RNN，一般RNN的size为(Batch, seq_len, channels)或者(seq_len, Batch, channels)，
        这里把seq_len放在channels后面，把所有时间步的数据拼起来，当做Conv1d的输入尺寸，实现卷积跨时间步的操作，
        很巧妙的设计。

        :param x: size of (Batch, input_channel, seq_len)
        :return: size of (Batch, output_channel, seq_len)
        """

        return self.network(x)

class TCN_V1(nn.Module):

    def __init__(self, input_size, output_size, num_channels,d_model=64,
                 kernel_size=[2,3,4], dropout=0.3, emb_dropout=0.1, tied_weights=False,shrink_flag=False):
        super(TCN_V1, self).__init__()
        # self.encoder = nn.Embedding(input_size, input_size)
        self.tcn1 = TemporalConvNet(input_size, num_channels[0], kernel_size[0], dropout=dropout,shrink_flag=False)
        self.tcn2 = TemporalConvNet(input_size, num_channels[1], kernel_size[1], dropout=dropout,shrink_flag=False)
        self.tcn3 = TemporalConvNet(input_size, num_channels[2], kernel_size[2], dropout=dropout,shrink_flag=False)


        self.decoder = nn.Linear(sum([i[-1] for i in num_channels]), output_size)
        if tied_weights:
            if num_channels[-1] != input_size:
                raise ValueError('When using the tied flag, nhid must be equal to emsize')
            self.decoder.weight = self.encoder.weight
            print("Weight tied")
        self.drop = nn.Dropout(emb_dropout)
        # self.feature_layer = get_torch_trans(heads=8, layers=1, channels=d_model)
        self.fc1 = nn.Linear(1,d_model)
        self.fc2 = nn.Linear(d_model,1)
        self.emb_dropout = emb_dropout
        # self.init_weights()

    def init_weights(self):
        # self.encoder.weight.data.normal_(0, 0.01)
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.normal_(0, 0.01)

    def forward(self, input):
        """Input ought to have dimension (B, C_in, L_in), where L_in is the seq_len; here the input is (B, L, C)"""
        # emb = self.drop(self.encoder(input))
        y1 = self.tcn1(input.transpose(1, 2)).transpose(1, 2)
        y2 = self.tcn2(input.transpose(1, 2)).transpose(1, 2)
        y3 = self.tcn3(input.transpose(1, 2)).transpose(1, 2)


        y= torch.cat([y1,y2,y3],dim=2)
        # y = self.forward_feature(y)
        y = self.decoder(y) #(B,L,output_size)
        return y[:,-1,:].contiguous()