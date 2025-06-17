import numpy as np

from src.tensor import Tensor

class CrossEntropy:
    def __init__(self):
        self.labels = None
        self.logits = None

    def forward(self, preds, labels):
        self.logits = preds
        self.labels = labels.data.astype(np.int32)

        self.logits.data -= np.max(self.logits.data, axis=1, keepdims=True)
        exps = np.exp(self.logits.data)

        self.probs = exps / np.sum(exps, axis=1, keepdims=True)

        batch_size = self.logits.data.shape[0]
        log_likelihood = -np.log(self.probs[np.arange(batch_size), self.labels] + 1e-9) # probs of true classes for each sample in batch
        loss = np.mean(log_likelihood)

        return Tensor(np.array(loss, dtype=np.float32), requires_grad=True, parent=self)

    def backward(self, grad):
        # d loss / d logits
        batch_size = self.logits.data.shape[0]
        res_grad = self.probs.copy()
        res_grad[np.arange(batch_size), self.labels] -= 1
        res_grad = res_grad / batch_size

        res_grad *= grad

        return [res_grad], [self.logits]
    
    def __call__(self, preds, labels):
        return self.forward(preds, labels)