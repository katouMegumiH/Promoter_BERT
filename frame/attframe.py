import logging
import math
from configs.configs import fixed_flag
import numpy as np
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
from model.bert_cnn import BertCNNModel
from frame.dataloaders import att_data_loader

Length = 80

def converts(temp):
    x_train = []
    for i in temp:
        lists = []
        lists[:0] = i
        lists = [int(i) for i in lists]
        x_train.append(lists)
    return x_train


# 数据框架
class AttFrame(nn.Module):
    def __init__(self, batch_size, lr, max_epoch,x_train,y_train,x_test,y_test,k,device):
        super().__init__()
        # 使用不同的模型，可以进一步封装到外层，使用参数控制，待修改
        self.model = BertCNNModel().to(device)
        self.is_bert = True
        # self.is_bert = False
        # 初始化设置各种参数，来自config文件
        self.batch_size = batch_size
        self.lr = lr
        self.max_epoch = max_epoch
        self.k = k
        self.device = device


        print("加载完成")
        self.fixed = fixed_flag
        if self.is_bert:
            self.train_loader = att_data_loader(x_train, y_train, shuffle=True, batch_size=self.batch_size,
                                                is_bert=True, is_fixed=self.fixed)
            self.test_loader = att_data_loader(x_test, y_test, shuffle=True, batch_size=self.batch_size, is_bert=True,
                                               is_fixed=self.fixed)
        else:
            self.train_loader = att_data_loader(x_train, y_train, shuffle=False, batch_size=self.batch_size,
                                                is_bert=False)
            self.test_loader = att_data_loader(x_test, y_test, shuffle=True, batch_size=self.batch_size, is_bert=False)

        self.loss_func = nn.BCELoss()#搭配sigmoid
        # self.loss_func = nn.CrossEntropyLoss()#搭配softmax
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.scheduler = ReduceLROnPlateau(self.optimizer, 'min')

    # 训练开始
    def train_start(self):

        best_f1, best_acc, best_recall, best_a = 1e-10, 1e-10, 1e-10, 1e-10
        best_fp, best_tp = 1e-10, 1e-10
        loss_log, acc_log, recall_log, sp_log, sn_log, mcc_log, f1_log, mcm_log = [], [], [], [], [], [], [], []
        best_pre_list, best_gold_list,best_ids_list = [], [], []

        for epoch in range(self.max_epoch):
            # Train
            self.model.train()
            train_loss = 0
            print(f"=== Epoch {epoch} train ===")
            t = tqdm(self.train_loader)
            # 把数据放进GPU
            is_test = False
            for data in t:
                if self.is_bert:
                    inputs, labels, masks, ids = data
                    masks = masks.to(self.device)
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)
                    out = self.model(inputs, attention_mask=masks)
                    # out = self.model(inputs, is_test)
                else:
                    inputs, labels = data
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)
                    out = self.model(inputs, is_test)
                    # out = self.model(inputs)
                loss = self.loss_func(out, labels.float())
                # loss = self.loss_func(out, labels.long())
                train_loss += loss.item()
                t.set_postfix(loss=loss)
                self.model.zero_grad()
                loss.backward()
                self.optimizer.step()
            avg_loss = train_loss / len(t)
            self.scheduler.step(avg_loss)
            # print(self.scheduler.get_lr()[0])
            loss_log.append(avg_loss)
            logging.info(f"Epoch: {epoch}, train loss: {avg_loss}")


            # 验证模型
            self.model.eval()
            t = tqdm(self.test_loader)
            pre_num, gold_num, correct_num = 1e-10, 1e-10, 1e-10
            # dev_losses = 0
            pre_list, gold_list = [], []
            old_pre_list = []
            old_ids_list = []
            ids_list = []
            pre_list2 = []
            with torch.no_grad():
                # attention 可视化
                # all_attention = []
                all_inputs = []
                all_labels = []
                all_out = []
                for iter_s, batch_samples in enumerate(t):
                    if self.is_bert:
                        inputs, labels, masks,ids = batch_samples
                        masks = masks.to(self.device)
                        inputs = inputs.to(self.device)
                        rel_out = self.model(inputs, attention_mask=masks)
                        # print(1)
                    else:
                        inputs, labels = batch_samples
                        inputs = inputs.to(self.device)
                        is_test = True
                        rel_out, attention = self.model(inputs, is_test)
                        # rel_out = self.model(inputs, is_test)
                    # 计算评价指标
                    labels = labels.numpy()
                    ids = np.array(ids)
                    rel_out = rel_out.to('cpu').numpy()
                    all_labels.append(labels)
                    all_out.append(rel_out)
                    # ——————————————————————————————————
                    idx = inputs.cpu().numpy()
                    all_inputs.append(idx)
                    # piece_attention = plot_attention(attention)
                    # for pre, gold, ids in zip(rel_out, labels, idx):  # , att, piece_attention
                    # for pre, gold, ids, att in zip(rel_out, labels, idx, piece_attention):
                    for pre, gold, id in zip(rel_out, labels, ids):
                        pre_set = np.round(pre)  # 取整
                        pre_set_1 = pre
                        old_pre_list.append(pre_set_1)
                        gold_set = gold
                        pre_list2.append(float(pre_set))
                        pre_list.append(float(pre))
                        gold_list.append(gold_set)
                        pre_num += 1
                        gold_num += 1

                        if pre_set == gold_set:
                            correct_num += 1
                            old_ids_list.append(id)
            print()
            print('loss:', avg_loss)
            print('正确个数', correct_num)
            print('预测个数', pre_num)
            acc_1 = correct_num / pre_num
            print("正确率是：", acc_1)


            tn, fp, fn, tp = confusion_matrix(gold_list, pre_list2).ravel()
            precision, recall = tp / (tp + fp), tp / (tp + fn)
            sp, sn = tn / (tn + fp), tp / (tp + fn)
            mcc = (tp * tn - tp * fn) / math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
            f1_score = 2 * precision * recall / (precision + recall)
            # acc_log.append(precision)
            acc_log.append(acc_1)
            # mcm_log.append(result)
            recall_log.append(recall)
            f1_log.append(f1_score)
            sp_log.append(sp)
            sn_log.append(sn)
            mcc_log.append(mcc)
            if best_a < acc_1:
                best_a = acc_1
                best_ids_list = list(set(old_ids_list))
            if best_f1 < f1_score:
                best_f1, best_acc, best_recall = f1_score, precision, recall
                # best_pre_list = old_pre_list
                best_pre_list = pre_list
                best_gold_list = gold_list

            true_pos = fp / (fp + fn)
            false_pos = tp / (tp + fn)
            if best_fp < false_pos and best_tp < true_pos:
                best_fp = false_pos
                best_tp = true_pos
            # print(f'true_num{tp},false_num{tn}')
            print('f1: %.4f, precision: %.4f, recall: %.4f' % (f1_score, precision, recall))
            print(f'best acc:{best_a}')
            print(f'best_tp,best_fp:{best_tp},{best_fp}')
            print(f'mcc:{mcc}')

        print(best_f1, best_acc, best_recall, best_a)
        best_sp = max(sp_log)
        best_sn = max(sn_log)
        mcc_log = [elem if not np.isnan(elem) else None for elem in mcc_log]
        while None in mcc_log:
            mcc_log.remove(None)

        best_mcc = max(mcc_log)


        save_metric_dict = {
            'loss': loss_log,
            'acc': acc_log,
            'recall': recall_log,
            'f1': f1_log,
            'sp': sp_log,
            'sn': sn_log,
            'mcc': mcc_log
            # 'mcm': mcm_log  # 多标签混淆矩阵 multilabel_confusion_matrix
        }
        # temp = []
        # temp.extend(best_gold_list)
        # temp.extend(best_pre_list)

        return best_a,best_mcc,best_sp,best_sn,best_f1,best_gold_list, best_pre_list,best_ids_list, save_metric_dict
        # with codecs.open('../' + name + 'roc_gold.json1', 'w', encoding='utf-8') as f:
        #     json.dump(best_gold_list, f, indent=4, ensure_ascii=False)
        # with codecs.open('../' + name + 'roc_pre.json1', 'w', encoding='utf-8') as f:
        #     json.dump(best_pre_list, f, indent=4, ensure_ascii=False)

# def plot_attention(attention):
#     # attention ==> batch_size, num layer,  sequence len,  sequence len
#     # 69,70 是剪切位点sas
#     # 138 个输入，每个输入 的 对应全句的attention系数
#     # 只需要剪切位点的两个 batch_size, num_layer,  2, 138
#     # attention = attention[:, :, 68:70, :]
#     attention = attention[:, :, 70:72, :]
#     # 合并多头, batch_size , 2, 138
#     attention = torch.mean(attention, 1)
#     # 合并剪切位点，mean一下，batch，138
#     attention = torch.mean(attention, 1)
#     # 单独拎出一个 1，138，138个gen对剪切位点的attention系数
#     pice_attention = attention.cpu().numpy()
#     # 可以做热度图，根据系数的大小，看看对剪切位点的影响
#     return pice_attention


def plot_roc(labels, predict_prob):
    false_positive_rate, true_positive_rate, thresholds = roc_curve(labels, predict_prob)
    roc_auc = auc(false_positive_rate, true_positive_rate)
    plt.title('ROC')
    plt.plot(false_positive_rate, true_positive_rate, 'b', label='AUC = %0.4f' % roc_auc)
    plt.legend(loc='lower right')
    # plt.plot([0, 1], [0, 1], 'r--')
    plt.ylabel('TPR')
    plt.xlabel('FPR')
    # plt.show()
    plt.savefig('roc.svg', dpi=1200)
    plt.savefig('roc.jpg')
