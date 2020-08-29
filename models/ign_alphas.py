import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
import copy


def _concat(xs):
    return torch.cat([x.view(-1) for x in xs])


class ign_alphas(object):

    def __init__(self, model, args):
        self.momentum = args.momentum
        self.weight_decay = args.weight_decay
        self.model = model
        self.v_model = copy.deepcopy(model)
        # self.alpha_optimizer = torch.optim.SGD(self.model._alphas_parameters(), lr=args.alphas_lr,
        #                                        momentum=args.alphas_momentum,
        #                                        weight_decay=args.alphas_weight_decay)
        self.alpha_optimizer = torch.optim.Adam(self.model._alphas_parameters(),lr=args.alphas_lr,betas=(0.5,0.999),weight_decay=0)
        self.isalpha = args.isalpha
        self.one_step_more = args.one_step_more
        if not self.one_step_more:
            print('no one step update in unrolled model')

    def _compute_unrolled_model(self, input, label, eta, optimizer, indices):
        if self.one_step_more:
            loss = self.model._loss(input, label, indices)
            theta = _concat(self.model.parameters()).data
            try:
                moment = _concat(optimizer.state[v]['momentum_buffer'] for v in self.model.parameters()).mul_(
                    self.momentum)
            except:
                moment = torch.zeros_like(theta)
            dtheta = _concat(torch.autograd.grad(loss, self.model.parameters())).data + self.weight_decay * theta
            # dalpha = _concat(torch.autograd.grad(loss, self.model._alphas_parameters())).data
            # print(torch.max(dalpha),torch.min(dalpha))

            unrolled_model = self._construct_model_from_theta(theta.sub(eta, moment + dtheta))
        else:
            theta = _concat(self.model.parameters()).data
            unrolled_model = self._construct_model_from_theta(theta)
        return unrolled_model

    def step(self, input_train, label_train, input_valid, label_valid, eta, optimizer, indices, eta_min=0):
        # eta = max(eta, eta_min)
        self.alpha_optimizer.zero_grad()
        # print(self.model.alphas[0:10])
        alphas_grad = self._backward_step_unrolled(input_train, label_train, input_valid, label_valid, eta, optimizer,
                                                   indices)
        # print(torch.max(alphas_grad),torch.min(alphas_grad))
        # print(self.model.alphas[0:10])
        if self.isalpha:
            self.alpha_optimizer.step()
        else:
            self.model._update_alphas(-alphas_grad)
            # print(torch.sum(self.model.alphas > 0) + torch.sum(self.model.alphas < 0))
        # print(self.model.alphas[0:10])
        return alphas_grad

    def _backward_step_unrolled(self, input_train, label_train, input_valid, label_valid, eta, optimizer,
                                indices):
        unrolled_model = self._compute_unrolled_model(input_train, label_train, eta, optimizer,
                                                      indices)  # build a new model with w'
        unrolled_loss = unrolled_model._loss(input_valid, label_valid)  # L_val(w',alpha)
        unrolled_loss.backward()
        # dalpha=0
        # dalpha = [v.grad for v in unrolled_model.alphas_parameters] # should be none
        dalpha = [torch.zeros(v.shape).cuda() for v in unrolled_model._alphas_parameters()]
        vector = [v.grad.data for v in unrolled_model.parameters()]  # dL_val(w',alpha)/dw'
        implicit_grads = self._hessian_vector_product(vector, input_train, label_train, indices)
        # print(torch.max(implicit_grads[0]),torch.min(implicit_grads[0]),torch.sum(implicit_grads[0]!=0))

        for g, ig in zip(dalpha, implicit_grads):
            g.data.sub_(eta, ig.data)

        for v, g in zip(self.model._alphas_parameters(), dalpha):
            if v.grad is None:
                v.grad = Variable(g.data)
            else:
                v.grad.data.copy_(g.data)

        return v.grad.data

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

    def _hessian_vector_product(self, vector, input, label, indices, r=1e-2):  # r=1e-2
        R = r / _concat(vector).norm()

        for p, v in zip(self.model.parameters(), vector):
            p.data.add_(R, v)

        loss = self.model._loss(input, label, indices)
        grads_p = torch.autograd.grad(loss, self.model._alphas_parameters())

        for p, v in zip(self.model.parameters(), vector):
            p.data.sub_(2 * R, v)

        loss = self.model._loss(input, label, indices)
        grads_n = torch.autograd.grad(loss, self.model._alphas_parameters())

        for p, v in zip(self.model.parameters(), vector):
            p.data.add_(R, v)

        return [(x - y).div_(2 * R) for x, y in zip(grads_p, grads_n)]

    # def _compute_hessian_vector(self,vector,input,label,indices):
    #     loss = self.model._loss(input, label, indices)
    #     loss.backward()
    #     vector1 = [v.grad.data for v in self.model.parameters()]
    #     pass
