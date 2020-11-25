import torch
import torch.nn as nn

from lib.basic.operations import ReLUConvBN, ReduceConv, OPS


class BasicRebuildCells(nn.Module):
    def __init__(self, genotype, c_prev_prev, c_prev, c, reduction_prev, reduction):
        super(BasicRebuildCells, self).__init__()

        self.pre_process0 = ReduceConv(c_prev_prev, c, 2) if reduction_prev else ReLUConvBN(c_prev_prev, c, 1, 1)
        self.pre_process1 = ReLUConvBN(c_prev, c, 1, 1)

        if reduction:
            ops, self.indices = zip(*genotype.reduce)
            self.concat = genotype.reduce_concat
        else:
            ops, self.indices = zip(*genotype.normal)
            self.concat = genotype.normal_concat

        self.steps = len(ops) // 2
        self.ops = nn.ModuleList(
            [OPS[name](c, 2 if reduction and index < 2 else 1) for name, index in zip(ops, self.indices)])

        self.out_channel = len(self.concat) * c

    def forward(self, s0, s1):
        s0 = self.pre_process0(s0)
        s1 = self.pre_process1(s1)

        states = [s0, s1]
        for i in range(self.steps):
            h1 = states[self.indices[2 * i]]
            h2 = states[self.indices[2 * i + 1]]
            op1 = self.ops[2 * i]
            op2 = self.ops[2 * i + 1]
            states.append(op1(h1) + op2(h2))
        return torch.cat([states[i] for i in self.concat], dim=1)


class BasicRebuildNetwork(nn.Module):
    def __init__(self, genotype, stem=16, layers=20, reduce=2):
        super(BasicRebuildNetwork, self).__init__()

        self.stem = nn.Sequential(nn.Conv2d(3, stem, 3, padding=1, bias=False), nn.BatchNorm2d(stem))

        cpp, cp, c, rp, r = stem, stem, stem, False, False
        reduce_layer = [layers // (reduce + 1) * index for index in range(1, reduce + 1)]
        self.cells = nn.ModuleList()
        for i in range(layers):
            if i in reduce_layer:
                c, r = 2 * c, True
            else:
                r = False
            self.cells.append(BasicRebuildCells(genotype, cpp, cp, c, rp, r))
            cpp, cp, rp = cp, self.cells[-1].out_channel, r

        self.out_channel = self.cells[-1].out_channel

    def forward(self, x):
        s0 = s1 = self.stem(x)
        for cell in self.cells:
            s0, s1 = s1, cell(s0, s1)
        return s1


if __name__ == "__main__":
    from torch.optim import Adam
    from lib.basic.genotypes import DARTS_V1

    dummy = torch.rand(7, 3, 32, 32).cuda()
    result = torch.rand(7, 256, 8, 8).cuda()

    model = BasicRebuildNetwork(DARTS_V1).cuda()
    optimizer = Adam(model.parameters())

    output = model(dummy)

    print(output.shape)
    print(model.out_channel)

    for _ in range(10):
        loss = torch.pow(model(dummy) - result, 2).sum()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(loss)
