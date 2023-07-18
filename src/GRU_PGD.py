import torch


class Gru_PGD():
    def __init__(self, model, emb_name, epsilon=1., eta=0.3):
        # emb_name这个参数要换成你模型中embedding的参数名
        self.model = model
        self.emb_name = emb_name
        self.epsilon = epsilon
        self.eta = eta
        self.emb_backup = {}
        self.grad_backup = {}

    def attack(self, is_first_attack=False):
        # print("self.emb_name==", self.model)
        # exit()
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if is_first_attack:
                    self.emb_backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0:
                    r_at = self.eta * param.grad / norm # 0.3 *param.grad/0.1822
                    param.data.add_(r_at)
                    #print("param.data_add",param.data.size()) # torch.Size([18359, 64])
                    #exit()
                    # param.data = self.project(name, param.data, self.epsilon)
        # exit()
    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.emb_name in name:
                assert name in self.emb_backup
                param.data = self.emb_backup[name]
        self.emb_backup = {}



    def project(self, param_name, param_data, epsilon):

        r = param_data - self.emb_backup[param_name]
        # print("r==",r.size())  #torch.Size([18359, 64])
        # print("norm(r)==",torch.norm(r)) # 0.300
        #exit()
        # 需要对r进行限制
        if torch.norm(r) > epsilon:
            r = epsilon * r / torch.norm(r)
        return self.emb_backup[param_name] + r

    def backup_grad(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                self.grad_backup[name] = param.grad.clone()


    def restore_grad(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.grad = self.grad_backup[name]


class Gru_PGD00():
    def __init__(self, model, emb_name, epsilon=1., eta=0.3):
        # emb_name这个参数要换成你模型中embedding的参数名
        self.model = model
        self.emb_name = emb_name
        self.epsilon = epsilon
        self.eta = eta
        self.emb_backup = {}
        self.grad_backup = {}

    def attack(self, is_first_attack=False):
        # print("self.emb_name==", self.model)
        # exit()
        for name, param in self.model.named_parameters():
            # for param in self.model.gru_layers.parameters():
            # print("self.name==", name)
            # print("param", param.requires_grad)
            if param.requires_grad and self.emb_name in name:
                # if param.requires_grad:
                if is_first_attack:
                    self.emb_backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                # print("norm==",norm)# tensor(0.1822)
                # print("eta",self.eta)
                # exit()
                if norm != 0:
                    r_at = self.eta * param.grad / norm  # 0.3 *param.grad/0.1822
                    param.data.add_(r_at)
                    # print("param.data_add",param.data.size()) # torch.Size([18359, 64])
                    # exit()
                    # param.data = self.project(name, param.data, self.epsilon)
        # exit()

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.emb_name in name:
                assert name in self.emb_backup
                param.data = self.emb_backup[name]
        self.emb_backup = {}

    def project(self, param_name, param_data, epsilon):

        r = param_data - self.emb_backup[param_name]
        # print("r==",r.size())  #torch.Size([18359, 64])
        # print("norm(r)==",torch.norm(r)) # 0.300
        # exit()
        # 需要对r进行限制
        if torch.norm(r) > epsilon:
            r = epsilon * r / torch.norm(r)
        return self.emb_backup[param_name] + r

    def backup_grad(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                self.grad_backup[name] = param.grad.clone()

    def restore_grad(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                param.grad = self.grad_backup[name]


####
if __name__=="__main__":
    pgd = PGD(model)
    K = 3
    for batch_input, batch_label in data:
        # 正常训练
        loss = model(batch_input, batch_label)
        loss.backward() # 反向传播，得到正常的grad
        pgd.backup_grad()
        # 对抗训练
        for t in range(K):
            pgd.attack(is_first_attack=(t==0)) # 在embedding上添加对抗扰动, first attack时备份param.data
            if t != K-1:
                model.zero_grad()
            else:
                pgd.restore_grad()
            loss_adv = model(batch_input, batch_label)
            loss_adv.backward() # 反向传播，并在正常的grad基础上，累加对抗训练的梯度
        pgd.restore() # 恢复embedding参数
        # 梯度下降，更新参数
        optimizer.step()
        model.zero_grad()

