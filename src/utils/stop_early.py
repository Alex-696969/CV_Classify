"""
Desc：如何连续多轮训练，best_acc都没有提升的话，就提前停止训练
"""

class Stop_early:
    def __init__(self, max_epoch:int = 20):
        self.max_epoch = max_epoch
        self.best_acc = 0
        self.is_stop = False


    def judge_stop(self, current_acc:float):
        count_epoch = 0
        if current_acc > self.best_acc:
            self.best_acc = current_acc
            count_epoch = 0
        else:
            count_epoch += 1
            if count_epoch == self.max_epoch:
                self.is_stop = True

