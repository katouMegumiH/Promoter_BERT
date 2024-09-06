import torch
import torch.nn as nn
from transformers import BertModel, BertConfig
import torch.nn.functional as F
from configs.configs import*


class BertCNNModel(nn.Module):
    def __init__(self):
        super(BertCNNModel, self).__init__()
        self.num_labels = 1
        self.hidden_size = 768
        model_config = BertConfig.from_pretrained(pretrain_path)
        # self.window_sizes = [3]#TATA
        # self.window_sizes = [2]
        # self.window_sizes = [1]
        self.window_sizes = [1,2,3,4,5,6]
        # self.window_sizes = [3, 4, 5, 6]
        # self.window_sizes = [3,4,5,6,7,8,9,10,11,12,13]
        # self.window_sizes = [3, 4, 5, 9, 10, 11, 16, 17, 21, 22, 26, 27, 32, 33, 37, 38]#NON_TATA
        # self.window_sizes = [3, 4, 5, 9, 10, 11, 13, 16, 17,19, 21, 22,24, 26, 27,29, 32, 33, 37, 38]
        self.max_text_len = max_length #TATA1_6
        # self.bert = RobertaModel(model_config)
        self.bert = BertModel(model_config)
        self.dropout = nn.Dropout(0.1)
        self.dropout_rate = 0.1
        # self.filter_size = 16#TATA
        self.filter_size = 64#NON_TATA
        # self.filter_size = 250
        # self.filter_size = 32
        # ##################################test
        # self.convs11 = nn.Conv1d(in_channels=self.hidden_size, out_channels=self.filter_size, kernel_size=11)
        # self.relu11 = nn.ReLU()
        # self.max_pool11 = nn.MaxPool1d(kernel_size=11)
        # ##################################test
        self.convs = nn.ModuleList([
            nn.Sequential(nn.Conv1d(in_channels=self.hidden_size,
                                    out_channels=self.filter_size,
                                    kernel_size=h),
                          # nn.BatchNorm1d(num_features=config.feature_size),
                          nn.ReLU(),
                          nn.MaxPool1d(kernel_size=self.max_text_len - h + 1))
            for h in self.window_sizes
        ])
        self.fc = nn.Linear(in_features=self.filter_size * len(self.window_sizes),
                            out_features=self.num_labels)

    def forward(self, inputs, token_type_ids=None, attention_mask=None, position_ids=None):
        outputs = self.bert(inputs, attention_mask=attention_mask)
        ##############test#################
        # pooled_output = outputs[0]
        # pooled_output = self.dropout(pooled_output)
        # pooled_output = pooled_output.view(pooled_output.size(0),-1)
        # # pooled_output = pooled_output.permute(0,2,1)
        # # pooled_output = pooled_output.view(-1,pooled_output.size(1))
        # logits = self.dense_1(pooled_output).squeeze(1)
        # out = logits.sigmoid()
        ##############test#################
        embed_x = outputs[0]
        embed_x = self.dropout(embed_x)
        embed_x = embed_x.permute(0, 2, 1)
        out = [conv(embed_x) for conv in self.convs]  # out[i]:batch_size x feature_size*1
        out = torch.cat(out, dim=1)  # 对应第二个维度（行）拼接起来，比如说5*2*1,5*3*1的拼接变成5*5*1
        out = out.view(-1, out.size(1))
        # out = out.view(-1, out.size(2))
        out = F.dropout(input=out, p=self.dropout_rate)
        out = self.fc(out).squeeze(1)  # 32 * 1
        out = out.sigmoid()  # 二分类

        return out


