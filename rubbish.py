# MGDA是什么？（False）
if self.MGDA:
    self.encoder = model[0]
    self.head = model[1]
    self.optimizer_encoder = optimizer[0]
    self.optimizer_head = optimizer[1]
    self.scheduler_encoder = torch.optim.lr_scheduler.MultiStepLR(
        self.optimizer_encoder, milestones=[50, 100, 150, 200], gamma=0.5)
    self.scheduler_head = torch.optim.lr_scheduler.MultiStepLR(
        self.optimizer_head, milestones=[50, 100, 150, 200], gamma=0.5)
    self.MGDA = config.MGDA
else:
    self.model = model
    self.kd_flag = kd_flag
    if self.kd_flag == 1:
        self.teacher = teacher
        for k, v in self.teacher.named_parameters():
            v.requires_grad = False  # fix parameters
    self.optimizer = optimizer
    self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[50, 100, 150, 200], gamma=0.5)