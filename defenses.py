import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from scipy.stats import norm
from statsmodels.stats.proportion import proportion_confint

def free_adv_train(model, data_tr, criterion, optimizer, lr_scheduler, \
                   eps, device, m=3, epochs=100, batch_size=128, dl_nw=10):
    """
    Free adversarial training, per Shafahi et al.'s work.
    Arguments:
    - model: randomly initialized model
    - data_tr: training dataset
    - criterion: loss function (e.g., nn.CrossEntropyLoss())
    - optimizer: optimizer to be used (e.g., SGD)
    - lr_scheduer: scheduler for updating the learning rate
    - eps: l_inf epsilon to defend against
    - device: device used for training
    - m: # times a batch is repeated
    - epochs: "virtual" number of epochs (equivalent to the number of 
        epochs of standard training)
    - batch_size: training batch_size
    - dl_nw: number of workers in data loader
    Returns:
    - trained model
    """
    # init data loader
    loader_tr = DataLoader(data_tr,
                           batch_size=batch_size,
                           shuffle=True,
                           pin_memory=True,
                           num_workers=dl_nw)
                           

    # init delta (adv. perturbation) - FILL ME
    delta = torch.zeros(data_tr[0][0].shape, requires_grad=True, device=device)

    # total number of updates - FILL ME
    total_num_iters = int(np.ceil(epochs / m))

    # when to update lr
    scheduler_step_iters = int(np.ceil(len(data_tr)/batch_size))

    # train - FILLE ME
    count = 0
    for i in range(total_num_iters):

        for j, data in enumerate(loader_tr, 0):
            # get inputs and labels
            inputs, labels = data[0].to(device), data[1].to(device)

            for k in range(m):
                # zero the parameter gradients
                optimizer.zero_grad()

                # Ascend on the global noise
                # noise_batch = torch.tensor(delta[0:inputs.size(0)], requires_grad=True).cuda()
                xadv = inputs + delta
                xadv.clamp_(0, 1.0)
                # in1.sub_(mean).div_(std).
                output = model(xadv)
                loss = criterion(output, labels)

                # compute gradient and do SGD step
                loss.backward()
                optimizer.step()

                pert = eps * torch.sign(delta.grad)
                # delta[0:inputs.size(0)] += pert.data
                delta = (delta + pert.data).clamp_(-eps, eps).clone().detach().to(device).requires_grad_(True)
                # delta.clamp_(-eps, eps)
                count += 1

                if count % scheduler_step_iters == 0:
                    lr_scheduler.step()

    
    # done
    return model


class SmoothedModel():
    """
    Use randomized smoothing to find L2 radius around sample x,
    s.t. the classification of x doesn't change within the L2 ball
    around x with probability >= 1-alpha.
    """

    ABSTAIN = -1

    def __init__(self, model, sigma):
        self.model = model
        self.sigma = sigma

    def _sample_under_noise(self, x, n, batch_size):
        """
        Classify input x under noise n times (with batch size 
        equal to batch_size) and return class counts (i.e., an
        array counting how many times each class was assigned the
        max confidence).
        """
        # FILL ME
        device = next(self.model.parameters()).device
        n_labels = 4
        frequencies = [0, 0, 0, 0]
        preds = None

        for i in range((n // batch_size) + 1):
            noise = torch.normal(0.0, std=self.sigma, size=[batch_size, x.shape[1], x.shape[2], x.shape[3]]).to(device)
            x_noisy = noise + x
            out = self.model(x_noisy)
            if preds is None:
                preds = out.argmax(dim=1)
            else:
                preds = torch.cat([preds, out.argmax(dim=1)])

        preds = preds[:n]
        for j in range(n_labels):
            frequencies[j] = (preds == j).sum().item()

        return frequencies
        
    def certify(self, x, n0, n, alpha, batch_size):
        """
        Arguments:
        - model: pytorch classification model (preferably, trained with
            Gaussian noise)
        - sigma: Gaussian noise's sigma, for randomized smoothing
        - x: (single) input sample to certify
        - n0: number of samples to find prediction
        - n: number of samples for radius certification
        - alpha: confidence level
        - batch_size: batch size to use for inference
        Outputs:
        - prediction / top class (ABSTAIN in case of abstaining)
        - certified radius (0. in case of abstaining)
        """
        
        # find prediction (top class c) - FILL ME
        counts = self._sample_under_noise(x, n0, batch_size)
        c = np.argmax(counts)
        counts = self._sample_under_noise(x, n, batch_size)


        # compute lower bound on p_c - FILL ME
        p_c = proportion_confint(counts[c], sum(counts), alpha)[0]
        if p_c > 0.5:
            radius = self.sigma * norm.ppf(p_c)
        else:
            c = SmoothedModel.ABSTAIN
            radius = 0

        # done
        return c, radius
        

class NeuralCleanse:
    """
    A method for detecting and reverse-engineering backdoors.
    """

    def __init__(self, model, dim=(1, 3, 32, 32), lambda_c=0.0005,
                 step_size=0.005, niters=2000):
        """
        Arguments:
        - model: model to test
        - dim: dimensionality of inputs, masks, and triggers
        - lambda_c: constant for balancing Neural Cleanse's objectives
            (l_class + lambda_c*mask_norm)
        - step_size: step size for SGD
        - niters: number of SGD iterations to find the mask and trigger
        """
        self.model = model
        self.dim = dim
        self.lambda_c = lambda_c
        self.niters = niters
        self.step_size = step_size
        self.loss_func = nn.CrossEntropyLoss()

    def find_candidate_backdoor(self, c_t, data_loader, device):
        """
        A method for finding a (potential) backdoor targeting class c_t.
        Arguments:
        - c_t: target class
        - data_loader: DataLoader for test data
        - device: device to run computation
        Outputs:
        - mask: 
        - trigger: 
        """
        # randomly initialize mask and trigger in [0,1] - FILL ME
        

        # run self.niters of SGD to find (potential) trigger and mask - FILL ME
        

        # done
        return mask, trigger
