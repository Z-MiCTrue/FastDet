from tqdm import tqdm
import torch
from torch import optim
from torchsummary import summary

from utils.datasets import collate_fn, TensorDataset
from utils.evaluation import CocoDetectionEvaluator
from module.loss import DetectorLoss
from module.FastestDet import FastestDet
from configs import Parameters


class FastestDet_Trainer:
    def __init__(self, cfg):
        # 配置文件
        self.cfg = cfg
        # 初始化模型结构
        if self.cfg.pretrained_weight is not None:
            print(f'load weight from: {self.cfg.pretrained_weight}')
            self.model = FastestDet(self.cfg.category_num, True).to(self.cfg.device)
            self.model.load_state_dict(torch.load(self.cfg.pretrained_weight))
        else:
            self.model = FastestDet(self.cfg.category_num, False).to(self.cfg.device)

        # 打印网络各层的张量维度
        summary(self.model, input_size=(3, self.cfg.input_height, self.cfg.input_width))

        # 构建优化器
        self.optimizer = optim.AdamW(params=self.model.parameters(),
                                     lr=self.cfg.learn_rate,
                                     weight_decay=self.cfg.weight_decay,
                                     )
        # 学习率衰减策略
        self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer,
                                                        milestones=self.cfg.milestones,
                                                        gamma=0.1)
        # 定义损失函数
        self.loss_function = DetectorLoss(self.cfg.device)
        # 定义验证函数
        self.evaluation = CocoDetectionEvaluator(self.cfg.names, self.cfg.device)
        # 定义历史最优mAP
        self.best_mAP = 0
        # 数据集加载
        val_dataset = TensorDataset(self.cfg.val_txt, False)
        train_dataset = TensorDataset(self.cfg.train_txt, False)
        # 验证集
        self.val_dataloader = torch.utils.data.DataLoader(val_dataset,
                                                          batch_size=self.cfg.batch_size,
                                                          shuffle=False,
                                                          collate_fn=collate_fn,
                                                          num_workers=self.cfg.num_workers)
        # 训练集
        self.train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                            batch_size=self.cfg.batch_size,
                                                            shuffle=True,
                                                            collate_fn=collate_fn,
                                                            num_workers=self.cfg.num_workers)

    def train(self):
        # 迭代训练
        self.model.train()
        print('Starting training for %g epochs...' % self.cfg.end_epoch)
        lr, iou, obj, cls, total = 0, 0, 0, 0, 0
        for epoch in range(self.cfg.end_epoch + 1):
            for img_batch, targets in tqdm(self.train_dataloader):
                # 数据预处理
                img_batch = img_batch.to(self.cfg.device).float() / 255
                targets = targets.to(self.cfg.device)
                # 模型推理
                preds = self.model.forward_train(img_batch)
                # loss计算
                iou, obj, cls, total = self.loss_function(preds, targets)
                # 反向传播求解梯度
                total.backward()
                # 更新模型参数
                self.optimizer.step()
                self.optimizer.zero_grad()
            # 模型验证及保存
            if epoch % 10 == 0 and epoch > 0:
                # 打印相关训练信息
                lr = self.optimizer.state_dict()['param_groups'][0]['lr']
                info = f'\n# --- Epoch-{epoch} lr: {lr} --- #\nIOU:{iou}; Obj:{obj}; Cls:{cls}; Total: {total}'
                tqdm.write(info)  # print(info)
                # 模型评估
                self.model.eval()
                tqdm.write('computer mAP: ')
                mAP05 = self.evaluation.compute_map(self.val_dataloader, self.model)
                if mAP05 > self.best_mAP:
                    torch.save(self.model.state_dict(),
                               f'./checkpoint/weight_best_AP05-{round(mAP05, 2)}_epoch-{epoch}.pth')
                    print(f'./checkpoint/weight_best_AP05-{round(mAP05, 2)}_epoch-{epoch}.pth has been saved over')
                    self.best_mAP = mAP05
                self.model.train()
            # 学习率调整
            self.scheduler.step()
        # 保存最后一次模型
        torch.save(self.model.state_dict(), f'./checkpoint/weight_last.pth')


if __name__ == "__main__":
    paras = Parameters()
    trainer = FastestDet_Trainer(paras)
    trainer.train()
