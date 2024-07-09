import numpy as np
import torch
import torchvision
import torch.optim as optim
from torch import nn
import myUtils
import copy
import time
from model import tree_TCN
from tqdm import tqdm
from myDataLoader import *
from sklearn import preprocessing
from treelib import Tree, Node
from pytorch_model_summary import summary

# additional subgradient descent on the sparsity-induced penalty term
def updateWN(my_model,tree, lambda_data=0.001):  # L1

    for count, m in enumerate(my_model.tree_nodes):
        if tree.contains(count):
            m.conv1.weight_g.grad.add_(lambda_data * torch.sign(m.conv1.weight_g.data))

def updateBN(my_model,tree, lambda_data=0.001):  # L1

    for count, m in enumerate(my_model.tree_nodes):
        if tree.contains(count):
            m.bn.weight.grad.add_(lambda_data * torch.sign(m.bn.weight.data))


def updateFC(my_model, len, lambda_data=0.001):  # L1

    my_model.linear2[len - 1].weight.grad.add_(lambda_data * torch.sign(my_model.linear2[len - 1].weight.data))


if __name__ == '__main__':
    torch.manual_seed(2)
    cuda_exist = torch.cuda.is_available()
    if cuda_exist:
        device = torch.device('cuda')
        print('cuda')
    else:
        device = torch.device('cpu')

    all_idx = np.arange(53)
    y_idx = np.array([29, 30,38]) - 1
    del_idx = np.array([45, 49, 52])
    covariates_idx = np.setdiff1d(all_idx, y_idx)
    covariates_idx = np.setdiff1d(covariates_idx, del_idx)

    tree, non_mask_ids = myUtils.creatTreeTE()
    tree_struct = Tree(tree=tree, deep=True)  # 树结构一直不变
    dic_father, dic_child = myUtils.father_childDict(tree)

    model = tree_TCN.TemporalConvNet_v1(num_inputs=50, num_outputs=len(y_idx), tree=tree_struct).cuda()
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    total_params = sum(p.numel() for p in model.parameters())
    print(total_params)

    dummy_input = torch.randn(32, 32, 50).cuda()


    f1_address = 'data_v2.xlsx'
    df = pd.read_excel(f1_address, header=0, index_col=0)
    initialData = df.values
    scaler = preprocessing.StandardScaler().fit(initialData)  # scaler保存方差和均值
    # np.savetxt("D:/PycharmProject/timeSeries/TE_results/mean.csv", scaler.mean_)
    # np.savetxt("D:/PycharmProject/timeSeries/TE_results/var.csv", scaler.var_)

    scorelabelData = preprocessing.scale(initialData)
    initTrainDataSet = scorelabelData[:4500, :]
    initTestDataSet = scorelabelData[4500:6000, :]

    trainSet = myAutoDataSet_v1(sw_width=32, n_out=1, covariates_idx=covariates_idx, y_idx=y_idx, initDataSet=initTrainDataSet)
    testSet = myAutoDataSet_v1(sw_width=32, n_out=1, covariates_idx=covariates_idx, y_idx=y_idx, initDataSet=initTestDataSet)

    trainLoader = DataLoader(trainSet, batch_size=32, shuffle=True, drop_last=True)
    testLoader = DataLoader(testSet, batch_size=32, drop_last=True)

    loss_fn = nn.MSELoss().to(device)
    loss_fn2 = nn.MSELoss(reduction='none').to(device)

    num_epochs = 80
    last_avg_loss = 1e5
    mask_ids = []
    best_mse = 1e6
    best_pred = []
    best_epoch = 0

    time_start_2 = time.clock()
    # code
    for epoch in range(num_epochs):

        tree_len = tree.size()
        # print(tree_len)
        avg_loss = 0
        total_time = 0
        model.train()
        with tqdm(trainLoader, mininterval=1.0, maxinterval=50.0) as it:
            it.set_description("train epoch %d" % epoch)
            for batch_idx, (X, Y) in enumerate(it, start=1):
                start = time.time()
                optimizer.zero_grad()
                X = X.to(device)
                Y = Y.to(device)
                y_pred = model(X, dic_child, dic_father,tree_len,mask_ids)
                loss = loss_fn(y_pred, Y[:, -1, :])
                loss.backward()
                updateWN(model,tree, lambda_data=0.001)
                # updateBN(model,tree, lambda_data=0.005)
                len_size = tree_len - len(mask_ids)
                updateFC(model, len_size, lambda_data=0.005)


                avg_loss += np.sqrt(loss.item())

                optimizer.step()

                # end = time.time()
                # total_time += end-start
                it.set_postfix(
                    ordered_dict={
                        "avg_epoch_loss": avg_loss / batch_idx,
                        "epoch": epoch,
                        # "time":total_time,
                    },
                    refresh=True,
                )

        #
        if epoch>60 and (epoch % 5 == 0 or epoch == num_epochs - 1):
            with torch.no_grad():
                model.eval()
                mse_total = 0
                mae_total = 0
                evalpoints_total = 0
                all_target = []
                all_pred = []
                with tqdm(testLoader, mininterval=5.0, maxinterval=50.0) as test:
                    test.set_description("test  epoch %d" % epoch)
                    for batch_idx, (X_test, Y_test) in enumerate(test, start=1):
                        X_test = X_test.to(device)
                        Y_test = Y_test.to(device)

                        y_test_pred = model(X_test,dic_child,dic_father,tree_len,mask_ids)
                        mse_current = loss_fn2(y_test_pred, Y_test[:, -1, :])
                        mae_current = torch.abs((y_test_pred - Y_test[:, -1, :]))
                        B, l, k = Y_test.shape
                        eval_points = torch.tensor(B * l * k)

                        mse_total += mse_current.sum().item()
                        mae_total += mae_current.sum().item()
                        evalpoints_total += eval_points

                        test.set_postfix(
                            ordered_dict={
                                "rmse_total": np.sqrt(mse_total / evalpoints_total),
                                "mae_total": mae_total / evalpoints_total,
                                "batch_idx": batch_idx,
                            },
                            refresh=True,
                        )
                        all_target.append(Y_test)
                        all_pred.append(y_test_pred.squeeze())
            all_target = torch.cat(all_target, dim=0)
            all_pred = torch.cat(all_pred, dim=0)
            if mse_total< best_mse:
                best_mse = mse_total
                best_pred = copy.deepcopy(all_pred)
                best_epoch = epoch
                best_model = model
                print(best_mse,epoch)
            # ev.multiplot(all_pred, all_target, epoch, nrows=len(y_idx), file_direct='plot', yButton=-3, yTop=3, plot_len=500)

        # 剪掉树节点
        if epoch %10 ==0 and epoch!=0 and epoch<=30 :
            fc_weight = model.linear2[tree_len - len(mask_ids) - 1].weight.data
            fc_weight_grad = model.linear2[tree_len - len(mask_ids) - 1].weight.grad
            fc_weight_abs = torch.abs(fc_weight)
            bias = model.linear2[tree_len - len(mask_ids) - 1].bias

            # OBD
            # grad_trans = fc_weight_grad.transpose(0, 1)
            # H = torch.mm(grad_trans, fc_weight_grad) / (tree_len - len(mask_ids))
            # weightindex = [fc_weight.squeeze()[i] ** 2 * H[i, i] for i in range(tree_len - len(mask_ids))]
            # weightindex_tensor = torch.tensor(weightindex).cuda()


            # threshold = 1e-13
            threshold = 0.0002

            # 使用 torch.lt() 函数进行逻辑比较，得到低于阈值的元素的布尔掩码
            # mask_bool = torch.lt(weightindex_tensor.squeeze(), threshold)
            mask_bool = torch.lt(fc_weight_abs.squeeze(), threshold)
            del_idx = torch.nonzero(mask_bool).squeeze()

            # sorted_weight_idx = sorted(range(fc_weight.shape[1]), key=lambda k: fc_weight[0,k], reverse=False)
            # del_idx = sorted_idx[:2]        #排序删除
            new_mask_ids = [non_mask_ids[i] for i in range(len(non_mask_ids)) if i in del_idx]
            non_mask_ids = [non_mask_ids[i] for i in range(len(non_mask_ids)) if i not in del_idx]
            mask_ids.extend(new_mask_ids)

            # mask = torch.ones_like(fc_weight.squeeze(), dtype=torch.bool).cuda()
            # mask[del_idx] = False
            mask_bool_not = torch.logical_not(mask_bool)
            weight_grad_prune = torch.masked_select(fc_weight_grad.squeeze(), mask_bool_not).unsqueeze(0)   #神经元剪枝
            weight_prune = torch.masked_select(fc_weight.squeeze(), mask_bool_not).unsqueeze(0)

            model.linear2[tree_len-len(mask_ids)-1].weight.data =  weight_prune
            model.linear2[tree_len-len(mask_ids)-1].bias.data =  bias.data
            print(mask_ids)

            tree, dic_father, dic_child, mask_ids = myUtils.treeStructUpdate(tree, mask_ids)
            print(mask_ids)
            print(non_mask_ids)

            model_summary = summary(model, dummy_input, dic_child, dic_father, tree.size(), mask_ids, show_input=False, show_hierarchical=False)
            print(model_summary)

        # prune channel
        #
        if epoch %20 ==0 and epoch>30:
            pruned = 0
            thre2 = 0.005 #WN
            # thre2 = 0.01
            index_mask = {} #tree中每个节点的channel_mask,key为tree的identifier
            for count,m in enumerate(model.tree_nodes):
                if tree.contains(count):
                    #改变tree_struct的data(channel)
                    node = tree_struct.get_node(count)
                    datalist = node.data

                    # WN
                    weight_g_copy = m.conv1.weight_g.data.abs().squeeze().clone()
                    weight_v_copy = m.conv1.weight_v.data.clone()
                    bias_copy = m.conv1.bias.data.clone()
                    mask_channel = weight_g_copy.gt(thre2).float().cuda()

                    # BN
                    # bn_weight_copy = m.bn.weight.data.abs().squeeze().clone()
                    # mask_channel = bn_weight_copy.gt(thre2).float().cuda()

                    remain_num = int(torch.sum(mask_channel))
                    if remain_num <=3:
                        #WN
                        median = torch.median(weight_g_copy)
                        mask_channel = weight_g_copy.gt(median).float().cuda()
                        #
                        # median = torch.median(bn_weight_copy)
                        # mask_channel = bn_weight_copy.gt(median).float().cuda()

                        remain_num = int(torch.sum(mask_channel))

                    pruned = pruned + mask_channel.shape[0] - torch.sum(mask_channel)
                    datalist[-1] = remain_num
                    index_mask[count] = mask_channel.clone()
            new_model =  tree_TCN.TemporalConvNet_v1(num_inputs=50,num_outputs=len(y_idx), tree=tree_struct).cuda()


            all_nodes = tree.all_nodes()
            node_ids = [node.identifier for node in all_nodes]
            node_ids_1= node_ids[1:] #去掉root
            for node_id in node_ids_1:
                father_id = dic_father[node_id]

                if father_id == 'root':
                    idx_father = np.array([i for i in range(len(covariates_idx) + len(y_idx))])
                else:
                    # np.squeeze 从数组的形状中删除单维度条目，即把shape中为1的维度去掉
                    # np.argwhere(a) 返回非0的数组元组的索引，其中a是要索引数组的条件
                    idx_father = np.squeeze(np.argwhere(np.asarray(index_mask[father_id].cpu().numpy())))

                idx1 = np.squeeze(np.argwhere(np.asarray(index_mask[node_id].cpu().numpy())))
                # new_model.tree_nodes[node_ids].conv1.weight_g.data
                if idx1.size == 1:
                    idx1 = np.resize(idx1, (1,))

                #wn层赋值
                # a = model.tree_nodes[node_id].conv1.weight_g.data
                new_model.tree_nodes[node_id].conv1.weight_g.data = model.tree_nodes[node_id].conv1.weight_g.data[idx1.tolist(), :, :].clone()
                new_model.tree_nodes[node_id].conv1.weight_v.data = model.tree_nodes[node_id].conv1.weight_v.data[idx1.tolist(),:, :][:,idx_father,:].clone()

                #BN赋值
                # new_model.tree_nodes[node_id].bn.weight.data = model.tree_nodes[node_id].bn.weight.data[idx1.tolist()].clone()
                # new_model.tree_nodes[node_id].bn.bias.data = model.tree_nodes[node_id].bn.bias.data[idx1.tolist()].clone()
                # new_model.tree_nodes[node_id].conv1.weight.data = model.tree_nodes[node_id].conv1.weight.data[idx1.tolist(), :, :][:,idx_father,:].clone()


                new_model.tree_nodes[node_id].conv1.bias.data = model.tree_nodes[node_id].conv1.bias.data[idx1.tolist()].clone()
                #con1x1赋值
                new_model.tree_nodes[node_id].downsample.weight.data = model.tree_nodes[node_id].downsample.weight.data[idx1.tolist()][:,idx_father,:].clone()
                new_model.tree_nodes[node_id].downsample.bias.data = model.tree_nodes[node_id].downsample.bias.data[idx1.tolist()].clone()
                #fc赋值
                new_model.tree_Linear[node_id].weight.data = model.tree_Linear[node_id].weight.data[:,idx1.tolist()].clone()
                new_model.tree_Linear[node_id].bias.data = model.tree_Linear[node_id].bias.data.clone()
                # m1.bias.data = m0.bias.data[idx1.tolist()].clone()
            #最后fc赋值
            middle_len = len(node_ids) - len(mask_ids)
            new_model.linear2[middle_len-1].weight.data = model.linear2[middle_len-1].weight.data.clone()
            new_model.linear2[middle_len-1].bias.data = model.linear2[middle_len-1].bias.data.clone()
            #root的线性层
            if tree.contains('root') and 'root' not in mask_ids:
                new_model.linear1.weight.data = model.linear1.weight.data.clone()
                new_model.linear1.bias.data = model.linear1.bias.data.clone()

            model = new_model
            optimizer = optim.SGD(model.parameters(), lr=0.1)

            print(node_ids)
            print(mask_ids)

            model_summary = summary(model, dummy_input, dic_child, dic_father, tree.size(), mask_ids, show_input=False, show_hierarchical=False)
            print(model_summary)

            # print(new_model)
    time_end_2 = time.clock()
    print("运行时间：" + str((time_end_2 - time_start_2) ) + "秒")
    # best_pred_numpy = best_pred.cuda().cpu().numpy()
    # all_target_numpy = torch.squeeze(all_target).cuda().cpu().numpy()
    # np.savetxt("D:/PycharmProject/timeSeries/TE_results/ptreetcn.csv", best_pred_numpy)
    # torch.save(best_model.state_dict(), 'model_params_TE.pth')


    model_summary = summary(model, dummy_input, dic_child, dic_father, tree.size(),mask_ids, show_input=False, show_hierarchical=False)
    print(model_summary)










