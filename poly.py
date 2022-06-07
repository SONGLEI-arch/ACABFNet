"""""
lr为新的学习率
base_lr为基准学习率
epoch为迭代次数
num_epoch为最大迭代次数
power控制曲线的形状（通常大于1）
"""""

def adjust_learning_rate_poly(optimizer, epoch, num_epochs, base_lr, power):
    lr = base_lr * (1-epoch/num_epochs)**power
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

