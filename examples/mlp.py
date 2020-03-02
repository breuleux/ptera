"""Simple MLP example.

Example usage:
  python mlp.py --dataroot ~/data
  python mlp.py --dataroot ~/data --lr 0.01 --actfn :torch.tanh

To see all options:
  python mlp.py -h

Any variable that is annotated with `cat.CliArgument` in a `@ptera` function
can be set on the command line. `ptera.auto_cli` will find them automatically.
"""

import os
from collections import deque

import torch
import torchvision

from ptera import Recurrence, auto_cli, cat, default, ptera
from ptera.torch import Grad


@ptera
def mnist():
    # Path to the directory where the datasets are
    dataroot: cat.CliArgument

    return torchvision.datasets.MNIST(
        root=os.path.expanduser(dataroot),
        train=True,
        transform=None,
        target_transform=None,
        download=True,
    )


@ptera
def layer(inp):
    W: cat.Learnable & cat.WeightMatrix
    b: cat.Learnable & cat.BiasVector
    actfn: cat.ActivationFunction

    act = torch.nn.functional.linear(inp, W, b)
    return actfn(act)


@ptera
def sequential(inp):
    layers: list

    h = Recurrence()
    h[0] = inp
    for i, layer in enumerate(layers):
        h[i + 1] = layer(h[i])
    return h[len(layers)]


@ptera
def step(inp, target):
    # How much to regulate the magnitude of the weights
    weight_reg: cat.CliArgument = default(0)

    model: cat.Model
    lossfn: cat.LossFunction

    if weight_reg:
        results = model.using(weights="$param:cat.WeightMatrix")(inp)
        output = results.value
        reg = sum(results.weights.map(lambda param: param.abs().sum()))
        loss = lossfn(output, target) + weight_reg * reg
    else:
        output = model(inp)
        loss = lossfn(output, target)

    return loss


def param(nin, nout, bias=False):
    if bias:
        return torch.nn.Parameter((torch.rand(1, nout) * 2 - 1) / (nin ** 0.5))
    else:
        return torch.nn.Parameter(
            (torch.rand(nout, nin) * 2 - 1) / (nin ** 0.5)
        )


@ptera
def make_network(layer_sizes):
    # Activation function for the network
    actfn: cat.CliArgument = default(torch.relu)

    layers = [
        layer.new(
            W=param(nin, nout), b=param(nin, nout, bias=True), actfn=actfn,
        )
        for nin, nout in zip(layer_sizes[:-1], layer_sizes[1:])
    ]
    layers[-1] = layers[-1].new(actfn=torch.nn.LogSoftmax(dim=1))
    network = step.new(
        lossfn=torch.nn.NLLLoss(), model=sequential.new(layers=layers)
    )
    return network


@ptera
def train():

    # Sizes of the hidden layers
    hidden: cat.CliArgument = default(1000)
    if isinstance(hidden, int):
        hidden = (hidden,)

    # Number of epochs
    epochs: cat.CliArgument & int = default(10)

    # Batch size
    batch_size: cat.CliArgument & int = default(32)

    # Learning rate
    lr: cat.CliArgument & float = default(0.1)

    # Seed
    seed: cat.CliArgument & int = default(1234)

    # Display weight statistics
    weight_stats: cat.CliArgument & bool = default(False)

    torch.random.manual_seed(seed)

    mn = mnist()
    train_data = mn.data
    train_targets = mn.targets
    train_data = train_data.reshape((-1, 784)) * (2.0 / 255) - 1.0

    nbatch = len(train_data) // batch_size
    running_losses = deque(maxlen=100)
    running_hits = deque(maxlen=100)
    layer_sizes = (784, *hidden, 10)

    my_step = make_network(layer_sizes).clone(return_object=True)

    @my_step.on("step{target} > output")
    def hits(output, target):
        return (output.max(dim=1).indices == target).sum()

    # @my_step.on(Grad("step{!!loss} >> $param:cat.Learnable"))
    # def update(param):
    #     param_value, param_grad = param
    #     param_value.data.sub_(lr * param_grad)

    if weight_stats:

        @my_step.on("$param:cat.WeightMatrix")
        def wstat(param):
            absw = param.abs()
            return absw.max(), absw.mean(), absw.min()

    for i in range(epochs):
        for j in range(nbatch):
            start = j * batch_size
            end = start + batch_size

            inp = train_data[start:end]
            tgt = train_targets[start:end]

            res = my_step(inp, tgt)
            running_losses.append(res.value)
            running_hits.append(int(sum(res.hits)) / batch_size)
            if j % 50 == 0:
                loss = sum(running_losses) / len(running_hits)
                accuracy = sum(running_hits) / len(running_hits)
                stats = [
                    f"E: {i + 1}/{epochs}",
                    f"B: {j + 1}/{nbatch}",
                    f"L: {loss:2.5f}",
                    f"A: {accuracy:.0%}",
                ]
                if weight_stats:
                    data = tuple(zip(*res.wstat))
                    mx = max(data[0])
                    avg = sum(data[1]) / len(data[1])
                    mn = min(data[2])
                    stats.append(f"W: {mx:.4f} > {avg:.4f} > {mn:.4f}")
                print(" -- ".join(stats))


if __name__ == "__main__":
    auto_cli(
        train, category=cat.CliArgument, eval_env=globals(), config_option=True
    )


def test_mlp():
    auto_cli(
        train,
        category=cat.CliArgument,
        eval_env=globals(),
        config_option=True,
        argv="--config config.json".split(),
    )
