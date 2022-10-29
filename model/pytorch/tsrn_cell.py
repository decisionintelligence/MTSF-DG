import numpy as np
import torch
from torch import nn,Tensor
import torch.nn.functional as F
from lib import utils

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class LayerParams:
    def __init__(self, rnn_network: torch.nn.Module, layer_type: str):
        self._rnn_network = rnn_network
        self._params_dict = {}
        self._biases_dict = {}
        self._type = layer_type

    def get_weights(self, shape):
        if shape not in self._params_dict:
            nn_param = torch.nn.Parameter(torch.empty(*shape, device=device))
            torch.nn.init.xavier_normal_(nn_param)
            self._params_dict[shape] = nn_param
            self._rnn_network.register_parameter('{}_weight_{}'.format(self._type, str(shape)),
                                                 nn_param)
        return self._params_dict[shape]

    def get_biases(self, length, bias_start=0.0):
        if length not in self._biases_dict:
            biases = torch.nn.Parameter(torch.empty(length, device=device))
            torch.nn.init.constant_(biases, bias_start)
            self._biases_dict[length] = biases
            self._rnn_network.register_parameter('{}_biases_{}'.format(self._type, str(length)),
                                                 biases)

        return self._biases_dict[length]

class TSRNCell(torch.nn.Module):
    def __init__(self, num_units, adj_mx0, adj_mx, max_diffusion_step, num_nodes, pre_k, pre_v, nonlinearity='tanh',
                 filter_type="laplacian", use_gc_for_ru=True, flag="encoder", gonv_random=None, layer=0):
        """

        :param num_units:
        :param adj_mx:
        :param max_diffusion_step:
        :param num_nodes:
        :param nonlinearity:
        :param filter_type: "laplacian", "random_walk", "dual_random_walk".
        :param use_gc_for_ru: whether to use Graph convolution to calculate the reset and update gates.
        """

        super().__init__()
        self._activation = torch.tanh if nonlinearity == 'tanh' else torch.relu
        self._num_nodes = num_nodes
        self._num_units = num_units
        self._max_diffusion_step = max_diffusion_step
        self._supports0 = []
        self._supports = []
        self._use_gc_for_ru = use_gc_for_ru
        self.pre_k = pre_k
        self.pre_v = pre_v
        self.layer = layer
        self.flag= flag
        supports = []
        if filter_type == "laplacian":
            supports.append(utils.calculate_scaled_laplacian(adj_mx, lambda_max=None))
        elif filter_type == "random_walk":
            supports.append(utils.calculate_random_walk_matrix(adj_mx).T)
        elif filter_type == "dual_random_walk":
            supports.append(utils.calculate_random_walk_matrix(adj_mx).T)
            supports.append(utils.calculate_random_walk_matrix(adj_mx.T).T)
        else:
            supports.append(utils.calculate_scaled_laplacian(adj_mx))
        for support in supports:
            self._supports.append(self._build_sparse_matrix(support))
        supports = []
        if type(None) != type(adj_mx0):
            if filter_type == "laplacian":
                supports.append(utils.calculate_scaled_laplacian(adj_mx0, lambda_max=None))
            elif filter_type == "random_walk":
                supports.append(utils.calculate_random_walk_matrix(adj_mx0).T)
            elif filter_type == "dual_random_walk":
                supports.append(utils.calculate_random_walk_matrix(adj_mx0).T)
                supports.append(utils.calculate_random_walk_matrix(adj_mx0.T).T)
            else:
                supports.append(utils.calculate_scaled_laplacian(adj_mx0))
            for support in supports:
                self._supports0.append(self._build_sparse_matrix(support))

        self._fc_params = LayerParams(self, 'fc')
        self._gconv_params0 = LayerParams(self, 'gconv0')
        self._gconv_params = LayerParams(self, 'gconv')
        self.W = nn.Parameter(torch.zeros(self._num_units, self._num_units), requires_grad=True)
        self.b = nn.Parameter(torch.zeros(num_nodes, self._num_units), requires_grad=True)
        self.R = nn.Parameter(torch.zeros(pre_k, num_nodes, self._num_units), requires_grad=True)
        self.attlinear = nn.Linear(2 * self._num_units, 1).to(device)
        self.gonv_random = gonv_random

    @staticmethod
    def _build_sparse_matrix(L):
        L = L.tocoo()
        indices = np.column_stack((L.row, L.col))
        indices = indices[np.lexsort((indices[:, 0], indices[:, 1]))]
        L = torch.sparse_coo_tensor(indices.T, L.data, L.shape, device=device)
        return L

    def forward(self, inputs, hx_k):
        """reasoning with Graph Convolution.
        :param inputs: (B, num_nodes * input_dim)
        :param hx_k: (B, pre_k, num_nodes, rnn_units)

        :return
        - Output: A `2-D` tensor with shape `(B, num_nodes * rnn_units)`.
        """
        bs, k, n, d = hx_k.shape
        preH = hx_k[:, -1:]
        for i in range(1, self.pre_v):
            preH = torch.cat([preH, hx_k[:, -(i + 1):-i]], -1)
        if self.pre_v != 0:
            preH = preH.reshape(bs, n, d * self.pre_v)
        else:
            preH = None
        output_size = self._num_units * (self.pre_v + 1)
        value = torch.sigmoid(self._fc(inputs, preH, output_size))
        value = torch.reshape(value, (-1, self._num_nodes, output_size))
        r, u = torch.split(tensor=value, split_size_or_sections=[self._num_units * self.pre_v, self._num_units], dim=-1)
        #r = r.repeat(1, 1, self.pre_v)

        convInput = F.leaky_relu_(self._gconv(inputs, r * preH, d, bias_start=0.0, flag_=self.pre_v))
        hx_k = hx_k.to(device)
        new_states = hx_k + self.R.unsqueeze(0).to(device)
        output = torch.matmul(convInput.to(device), self.W.to(device)) + self.b.unsqueeze(0).to(device)
        if self._activation is not None:
            output = self._activation(output)
        output = (1.0 - u) * output + u * self.reasoning(new_states, convInput).to(device)
        output = output.unsqueeze(1)

        x = hx_k[:, 1:k]
        hx_k = torch.cat([x, output], dim=1)
        output = output.reshape(bs, n * d)
        return output, hx_k

    @staticmethod
    def _concat(x, x_):
        x_ = x_.unsqueeze(0)
        return torch.cat([x, x_], dim=0)

    def _fc(self, inputs, state, output_size, bias_start=0.0):
        batch_size = inputs.shape[0]
        inputs = torch.reshape(inputs, (batch_size * self._num_nodes, -1))
        state = torch.reshape(state, (batch_size * self._num_nodes, -1))
        inputs_and_state = torch.cat([inputs, state], dim=-1)
        input_size = inputs_and_state.shape[-1]
        weights = self._fc_params.get_weights((input_size, output_size))
        value = torch.matmul(inputs_and_state, weights)
        biases = self._fc_params.get_biases(output_size, bias_start)
        value += biases
        return value

    def _gconv(self, inputs, state, output_size, bias_start=0.0, flag_=0):
        # input / state: (batch_size, num_nodes, input_dim/state_dim)
        batch_size = inputs.shape[0]
        inputs = torch.reshape(inputs, (batch_size, self._num_nodes, -1))
        if flag_ == 0:
            inputs_and_state = inputs
        else:
            state = torch.reshape(state, (batch_size, self._num_nodes, flag_, -1))
            statex, statey =  torch.split(tensor=state, split_size_or_sections=[self._num_units//2, self._num_units//2], dim=-1)
            inx, iny = torch.split(tensor=inputs, split_size_or_sections=[self._num_units//2, self._num_units//2], dim=-1)
            statex = torch.reshape(statex, (batch_size, self._num_nodes, -1))
            statey = torch.reshape(statey, (batch_size, self._num_nodes, -1))
            inputs_and_statex = torch.cat([inx, statex], dim=2)
            inputs_and_statey = torch.cat([iny, statey], dim=2)
        
        if self.layer == 0:
        #self.flag == "encoder" and 
            gonv_random = self.gonv_random.repeat(batch_size, 1, 1)
            inputs_and_statex = torch.cat([inputs_and_statex, gonv_random], dim=2)
            inputs_and_statey = torch.cat([inputs_and_statey, gonv_random], dim=2)


        input_size = inputs_and_statex.size(2)

        x = inputs_and_statex
        x0 = x.permute(1, 2, 0)  # (num_nodes, total_arg_size, batch_size)
        x0 = torch.reshape(x0, shape=[self._num_nodes, input_size * batch_size])
        x = torch.unsqueeze(x0, 0)

        if self._max_diffusion_step == 0:
            pass
        else:
            for support in self._supports:
                x1 = torch.sparse.mm(support, x0)
                x = self._concat(x, x1)
                for k in range(2, self._max_diffusion_step + 1):
                    x2 = 2 * torch.sparse.mm(support, x1) - x0
                    x = self._concat(x, x2)
                    x1, x0 = x2, x1
            if len(self._supports0) != 0:
                y = inputs_and_statey
                y0 = y.permute(1, 2, 0)  # (num_nodes, total_arg_size, batch_size)
                y0 = torch.reshape(y0, shape=[self._num_nodes, input_size * batch_size])
                y = torch.unsqueeze(y0, 0)
                for support in self._supports0:
                    y1 = torch.sparse.mm(support, y0)
                    y = self._concat(y, y1)
                    for k in range(2, self._max_diffusion_step + 1):
                        y2 = 2 * torch.sparse.mm(support, y1) - y0
                        y = self._concat(y, y2)
                        y1, y0 = y2, y1

        num_matrices = len(self._supports) * self._max_diffusion_step + 1
        x = torch.reshape(x, shape=[num_matrices, self._num_nodes, input_size, batch_size])
        x = x.permute(3, 1, 2, 0)
        x = torch.reshape(x, shape=[batch_size * self._num_nodes, input_size * num_matrices])
        if len(self._supports0) != 0:
            y = torch.reshape(y, shape=[num_matrices, self._num_nodes, input_size, batch_size])
            y = y.permute(3, 1, 2, 0)
            y = torch.reshape(y, shape=[batch_size * self._num_nodes, input_size * num_matrices])
            yweights = self._gconv_params0.get_weights((input_size * num_matrices, int(output_size/2)))
            xweights = self._gconv_params.get_weights((input_size * num_matrices, int(output_size/2)))
            x = torch.matmul(x, xweights)
            y = torch.matmul(y, yweights)
            x = torch.cat([x, y], dim=-1)
        else:
            weights = self._gconv_params.get_weights((input_size * num_matrices, output_size))
            x = torch.matmul(x, weights)

        biases = self._gconv_params.get_biases(output_size, bias_start)
        x += biases
        return torch.reshape(x, [batch_size, self._num_nodes, output_size])

    def reasoning(self, inputs: Tensor, cur: Tensor):
        bs, k, n, d = inputs.size()
        cur = torch.unsqueeze(cur, 1)
        cur = cur.repeat(1, k, 1, 1)
        #x = inputs.reshape(bs, k, -1)
        x = torch.cat([inputs, cur], dim=3)
        x = x.permute(0, 2, 1, 3)
        x = x.to(device)
        out = self.attlinear(x).to(device)
        weight = F.softmax(out, dim=-2).to(device)
        weight = weight.permute(0, 2, 1, 3)
        outputs = (inputs * weight).sum(dim=1).reshape(bs, n, d).to(device)
        return outputs
