import torch
from torch import nn
import torch.nn.functional as f

from lib.basic.operations import ReLUConvBN, ReduceConv, OPS, PRIMITIVES
from lib.basic.genotypes import Genotype


class MixedOp(nn.Module):
    def __init__(self, channel, stride):
        super(MixedOp, self).__init__()

        self.operations = nn.ModuleList([OPS[primitive](channel, stride) for primitive in PRIMITIVES])

    def forward(self, x, weights):
        return sum(w * op(x) for w, op in zip(weights, self.operations))


class BasicSearchCells(nn.Module):
    def __init__(self, c_prev_prev, c_prev, c, reduction_prev, reduction, steps=4, multiplier=4):
        super(BasicSearchCells, self).__init__()

        self.reduction = reduction
        self.steps = steps
        self.multiplier = multiplier

        self.pre_process0 = ReduceConv(c_prev_prev, c, 2) if reduction_prev else ReLUConvBN(c_prev_prev, c, 1, 1)
        self.pre_process1 = ReLUConvBN(c_prev, c, 1, 1)

        self.operations = nn.ModuleList(
            [MixedOp(c, 2 if reduction and j < 2 else 1) for i in range(steps) for j in range(2 + i)])

    def forward(self, s0, s1, weights):
        states = [self.pre_process0(s0), self.pre_process1(s1)]

        for i in range(self.steps):
            offset = i * (i + 3) // 2
            s = sum(self.operations[offset + j](state, weights[offset + j]) for j, state in enumerate(states))
            states.append(s)

        return torch.cat(states[-self.multiplier:], dim=1)


class BasicSearchNetwork(nn.Module):
    def __init__(self, mode, stem=16, steps=4, multiplier=4, layers=8, reduce=2):
        super(BasicSearchNetwork, self).__init__()

        assert mode in ["softmax", "sigmoid", "gumbel"]

        self.mode = mode
        self.step = steps
        self.multiplier = multiplier

        self.stem = nn.Sequential(nn.Conv2d(3, stem, 3, padding=1, bias=False), nn.BatchNorm2d(stem))

        cpp, cp, c, rp, r = stem, stem, stem, False, False
        reduce_layer = [layers // (reduce + 1) * index for index in range(1, reduce + 1)]
        self.cells = nn.ModuleList()
        for i in range(layers):
            if i in reduce_layer:
                c, r = 2 * c, True
            else:
                r = False
            self.cells.append(BasicSearchCells(cpp, cp, c, rp, r, steps, multiplier))
            cpp, cp, rp = cp, multiplier * c, r
        self.out_channel = cp

        ops_length = len(PRIMITIVES)
        path_length = steps * (steps + 3) // 2

        self.alpha_normal = nn.Parameter(1e-3 * torch.randn(path_length, ops_length))
        self.alpha_reduce = nn.Parameter(1e-3 * torch.randn(path_length, ops_length))

    def forward(self, x, tau=1):
        weight_normal, weight_reduce = self.get_weight(tau)
        s0 = s1 = self.stem(x)
        for cell in self.cells:
            s0, s1 = s1, cell(s0, s1, weight_reduce if cell.reduction else weight_normal)
        return s1

    def get_weight(self, tau):
        if self.mode == "softmax":
            weight_normal = f.softmax(self.alpha_normal / tau, dim=-1)
            weight_reduce = f.softmax(self.alpha_reduce / tau, dim=-1)
        elif self.mode == "sigmoid":
            weight_normal = f.sigmoid(self.alpha_normal / tau)
            weight_reduce = f.sigmoid(self.alpha_reduce / tau)
        elif self.mode == "sigmoid":
            weight_normal = f.gumbel_softmax(self.alpha_normal, tau, dim=-1)
            weight_reduce = f.gumbel_softmax(self.alpha_reduce, tau, dim=-1)
        else:
            raise NotImplementedError(f"{self.mode} not implemented.")

        return weight_normal, weight_reduce

    def genotype(self):
        def parse(weights, top_k=2):
            genotype = []
            for i in range(self.step):
                offset = i * (i + 3) // 2
                weight = weights[offset:offset + i + 2]
                edges = []
                for step, weight in enumerate(weight):
                    value, index = torch.max(weight, dim=-1)
                    edges.append((value.item(), step, index.item()))
                top_edge = sorted(edges, key=lambda item: item[0], reverse=True)[:top_k]
                genotype.extend([(PRIMITIVES[index], step) for _, step, index in top_edge])
            return genotype

        weight_normal = parse(self.alpha_normal.data)
        weight_reduce = parse(self.alpha_reduce.data)
        concat = range(2 + self.step - self.multiplier, self.step + 2)

        return Genotype(normal=weight_normal, normal_concat=concat, reduce=weight_reduce, reduce_concat=concat)


if __name__ == "__main__":
    from torch.optim import Adam

    dummy = torch.rand(7, 3, 32, 32).cuda()
    result = torch.rand(7, 256, 8, 8).cuda()

    model = BasicSearchNetwork("softmax").cuda()
    optimizer = Adam(model.parameters())

    output = model(dummy)

    print(output.shape)
    print(model.genotype())

    for _ in range(10):
        loss = torch.pow(model(dummy) - result, 2).sum()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(loss)
        print(model.genotype())
