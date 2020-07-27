import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Variable


def _concat(xs):
    return torch.cat([x.view(-1) for x in xs])


class ign_alphas(object):

    def __init__(self, model, args):
        self.alphas_momentum = args.momentum
        self.alphas_weight_decay = args.weight_decay
        self.model = model
        self.optimizer = torch.optim.Adam(self.model.alphas_parameters, lr=args.alphas_lr, betas=(0.5, 0.999),
                                          weight_decay=args.alphas_weight_decay)

    def _compute_unrolled_model(self, input, label, eta, optimizer, indices):
        loss = self.model._loss(input, label, indices)
        theta = _concat(self.model.parameters()).data
        try:
            moment = _concat(optimizer.state[v]['momentum_buffer'] for v in self.model.parameters()).mul_(
                self.alphas_momentum)
        except:
            moment = torch.zeros_like(theta)
        dtheta = _concat(torch.autograd.grad(loss, self.model.parameters())).data + self.alphas_weight_decay * theta
        unrolled_model = self._construct_model_from_theta(theta.sub(eta, moment + dtheta))
        return unrolled_model

    def step(self, input_train, label_train, input_valid, label_valid, eta, optimizer, indices):
        self.optimizer.zero_grad()
        # print(self.model.alphas[0:10])
        self._backward_step_unrolled(input_train, label_train, input_valid, label_valid, eta, optimizer, indices)
        # print(self.model.alphas[0:10])
        self.optimizer.step()
        # print(self.model.alphas[0:10])

    def _backward_step_unrolled(self, input_train, label_train, input_valid, label_valid, eta, optimizer,
                                indices):
        unrolled_model = self._compute_unrolled_model(input_train, label_train, eta, optimizer,
                                                      indices)  # build a new model with w'
        unrolled_loss = unrolled_model._loss(input_valid, label_valid)  # L_val(w',alpha)
        unrolled_loss.backward()
        # dalpha=0
        # dalpha = [v.grad for v in unrolled_model.alphas_parameters] # should be none
        dalpha=[torch.zeros(v.shape).cuda() for v in unrolled_model.alphas_parameters]
        vector = [v.grad.data for v in unrolled_model.parameters()]  # dL_val(w',alpha)/dw'
        implicit_grads = self._hessian_vector_product(vector, input_train, label_train, indices)
        # print(torch.max(implicit_grads[0]),torch.min(implicit_grads[0]),torch.sum(implicit_grads[0]!=0))


        for g, ig in zip(dalpha, implicit_grads):
            g.data.sub_(eta, ig.data)

        for v, g in zip(self.model.alphas_parameters, dalpha):
            if v.grad is None:
                v.grad = Variable(g.data)
            else:
                v.grad.data.copy_(g.data)

    def _construct_model_from_theta(self, theta):
        model_new = self.model.new()
        model_dict = self.model.state_dict()
        params, offset = {}, 0
        for k, v in self.model.named_parameters():
            v_length = np.prod(v.size())
            params[k] = theta[offset:offset + v_length].view(v.size())
            offset += v_length

        assert offset == len(theta)
        model_dict.update(params)
        model_new.load_state_dict(model_dict)
        return model_new.cuda()

    def _hessian_vector_product(self, vector, input, label, indices, r=1e-2): #r=1e-2
        R = r / _concat(vector).norm()


        for p, v in zip(self.model.parameters(), vector):
            p.data.add_(R, v)


        loss = self.model._loss(input, label, indices)
        grads_p = torch.autograd.grad(loss, self.model.alphas_parameters)

        for p, v in zip(self.model.parameters(), vector):
            p.data.sub_(2 * R, v)


        loss = self.model._loss(input, label, indices)
        grads_n = torch.autograd.grad(loss, self.model.alphas_parameters)

        for p, v in zip(self.model.parameters(), vector):
            p.data.add_(R, v)

        return [(x - y).div_(2 * R) for x, y in zip(grads_p, grads_n)]
