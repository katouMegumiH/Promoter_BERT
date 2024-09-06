import codecs
import json
import random
import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve, auc
from configs.configs import *
from frame.attframe import AttFrame


def seed_torch(m_seed=2021):
    random.seed(m_seed)
    np.random.seed(m_seed)
    torch.manual_seed(m_seed)

def roc_data(n_splits,gold_list,pre_list,dataset):
    for j in range(n_splits):
        df = pd.DataFrame()
        df['label'] = gold_list[j]
        df['pre'] = pre_list[j]
        df.to_csv('./result/{}/roc/{}-fold_roc.csv'.format(dataset, j + 1),index=False)


def roc(y_true,y_score,n,dataset):
    fpr, tpr, thresholds = roc_curve(y_true,y_score)
    roc_auc = auc(fpr, tpr)
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, label=u'{}-fold (AUC = %0.3f)'.format(n+1)% roc_auc)

    plt.legend(loc='lower right')
    plt.plot([0,1],[0,1],'r--')
    plt.xlim([-0.1,1.1])
    plt.ylim([-0.1,1.1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.grid(linestyle='-.')
    plt.grid(True)
    # plt.show()
    # plt.savefig('./result/{}/{}/roc/roc.jpg'.format(p,model))
    plt.savefig('./result/{}/roc/roc.jpg'.format(dataset))




if __name__ == '__main__':
    # 设置参数
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_torch(seed)
    #数据读取与10折
    n_splits = 10
    dataset = 'TATA'# NON_TATA,TATA,group
    #结果list
    lacc, lmcc, lsp, lsn,lf1, gold_list, pre_list,metrics = [], [], [], [], [], [], [], []
    #k折
    for k in range(n_splits):
        print('-------第{}折-------'.format(k))
        x_train,y_train,x_test,y_test = [],[],[],[]
        df_train = pd.read_csv(f'./data/{dataset}/{k+1}-fold_train.csv')
        df_test = pd.read_csv(f'./data/{dataset}/{k+1}-fold_test.csv')
        for i in df_train['seq']:
            temp = i.split('/')
            x_train.append(temp)
        for j in df_train['label']:
            y_train.append(j)
        for i in df_test['seq']:
            temp = i.split('/')
            x_test.append(temp)
        for j in df_test['label']:
            y_test.append(j)
        y_train = np.array(y_train)
        y_test = np.array(y_test)
        # frame
        framework = AttFrame(batch_size, lr, epoch,x_train,y_train,x_test,y_test,k,device)
        best_acc,best_mcc,best_sp,best_sn,best_f1,best_gold_list, best_pre_list,gooddata_list, save_metric_dict = framework.train_start()

        lacc.append(best_acc)
        lmcc.append(best_mcc)
        lsp.append(best_sp)
        lsn.append(best_sn)
        lf1.append(best_f1)
        gold_list.append(best_gold_list)
        pre_list.append(best_pre_list)
        metrics.append(save_metric_dict)


    mean_acc = float(np.mean(lacc))
    print('----------------mean_acc=', mean_acc, '------------------')
    acc = str(np.mean(lacc)) + '±' + str(np.var(lacc))
    mcc = str(np.mean(lmcc)) + '±' + str(np.var(lmcc))
    sp = str(np.mean(lsp)) + '±' + str(np.var(lsp))
    sn = str(np.mean(lsn)) + '±' + str(np.var(lsn))
    f1 = str(np.mean(lf1)) + '±' + str(np.var(lf1))
    metric_dict = {
        'acc': acc,
        'f1':f1,
        'mcc': mcc,
        'sp': sp,
        'sn': sn,
        'lacc':lacc,
        'lf1':lf1,
        'lmcc':lmcc,
        'lsp':lsp,
        'lsn':lsn
    }

    with codecs.open('./result/{}/'.format(dataset) + 'metric.json', 'w', encoding='utf-8') as f:
        json.dump(metric_dict, f, indent=4, ensure_ascii=False)
    for n in range(len(metrics)):
        with codecs.open('./result/{}/'.format(dataset) + '{}-fold_metric.json'.format(n+1), 'w', encoding='utf-8') as f:
            json.dump(metrics[n], f, indent=4, ensure_ascii=False)
    #roc
    roc_data(n_splits, gold_list, pre_list, dataset)
    for m in range(n_splits):
        dfroc = pd.read_csv('./result/{}/roc/{}-fold_roc.csv'.format(dataset,m+1))
        y_true = dfroc['label']
        y_score = dfroc['pre']
        roc(y_true, y_score, m,dataset)




