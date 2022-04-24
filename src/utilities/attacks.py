# mypy: ignore-errors
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

# adapted from https://github.com/mnmueller/eran_vnncomp2021/blob/master/tf_verify/attacks.py
# commit hash: a8eae0e1e6e26081cdc9f57747c333630f04807a


def margin_loss(logits, y):
    logit_org = logits.gather(1, y.view(-1, 1))
    logit_target = logits.gather(
        1, (logits - torch.eye(10)[y].to("cuda") * 9999).argmax(1, keepdim=True)
    )
    loss = -logit_org + logit_target
    loss = torch.sum(loss)
    return loss


def constraint_loss(logits, constraints, and_idx=None):
    loss = 0
    for i, or_list in enumerate(constraints):
        or_loss = 0
        for cstr in or_list:
            if cstr[0] == -1:
                or_loss += -logits[:, cstr[1]]
            elif cstr[1] == -1:
                or_loss += logits[:, cstr[0]]
            else:
                or_loss += logits[:, cstr[0]] - logits[:, cstr[1]]
        if and_idx is not None:
            loss += torch.where(and_idx == i, or_loss, torch.zeros_like(or_loss))
        else:
            loss += or_loss
    return -loss


class step_lr_scheduler:
    def __init__(self, initial_step_size, gamma=0.1, interval=10):
        self.initial_step_size = initial_step_size
        self.gamma = gamma
        self.interval = interval
        self.current_step = 0

    def step(self, k=1):
        self.current_step += k

    def get_lr(self):
        if isinstance(self.interval, int):
            return self.initial_step_size * self.gamma ** (
                np.floor(self.current_step / self.interval)
            )
        else:
            phase = len([x for x in self.interval if self.current_step >= x])
            return self.initial_step_size * self.gamma ** (phase)


def torch_whitebox_attack(
    model,
    device,
    sample,
    constraints,
    specLB,
    specUB,
    input_nchw=True,
    restarts=1,
    stop_early=True,
    ODI=True,
):
    if restarts == 0:
        return None, sample.detach().cpu().numpy()
    input_shape = list(sample.shape)
    input_shape = ([1] if len(input_shape) in [3, 1] else []) + input_shape
    nhwc_shape = (
        input_shape[0:1] + input_shape[-2:] + input_shape[-3:-2]
        if input_nchw
        else input_shape
    )
    nchw_shape = (
        input_shape
        if input_nchw
        else input_shape[0:1] + input_shape[-1:] + input_shape[-3:-1]
    )
    specLB_t = specLB.reshape(nchw_shape if input_nchw else nhwc_shape).clone().detach()
    specUB_t = specUB.reshape(nchw_shape if input_nchw else nhwc_shape).clone().detach()
    sample = sample.reshape(input_shape if input_nchw else nhwc_shape)
    if len(input_shape) == 4:
        specLB_t = specLB_t.permute((0, 1, 2, 3) if input_nchw else (0, 3, 1, 2)).to(
            device
        )
        specUB_t = specUB_t.permute((0, 1, 2, 3) if input_nchw else (0, 3, 1, 2)).to(
            device
        )
        sample = sample.permute((0, 1, 2, 3) if input_nchw else (0, 3, 1, 2))
    X = Variable(sample, requires_grad=True).to(device)

    if np.prod(input_shape) < 10 or not ODI:
        ODI_num_steps = 0
    else:
        ODI_num_steps = 10

    adex, worst_x = _pgd_whitebox(
        model,
        X,
        constraints,
        specLB_t,
        specUB_t,
        device,
        lossFunc="GAMA",
        restarts=restarts,
        ODI_num_steps=ODI_num_steps,
        stop_early=stop_early,
    )
    if adex is None:
        adex, _ = _pgd_whitebox(
            model,
            X,
            constraints,
            specLB_t,
            specUB_t,
            device,
            lossFunc="margin",
            restarts=restarts,
            ODI_num_steps=ODI_num_steps,
            stop_early=stop_early,
        )
    if len(input_shape) == 4:
        if adex is not None:
            adex = [adex[0][0].transpose((0, 1, 2) if input_nchw else (1, 2, 0))]
        if worst_x is not None:
            worst_x = worst_x.transpose((0, 1, 2) if input_nchw else (1, 2, 0))

    if adex is not None:
        assert (adex[0] >= specLB.cpu().numpy()).all() and (
            adex[0] <= specUB.cpu().numpy()
        ).all()
        print("Adex found via attack")
    else:
        assert (worst_x >= specLB.cpu().numpy()).all() and (
            worst_x <= specUB.cpu().numpy()
        ).all()
        print("No adex found via attack")
    return adex, worst_x


def _evaluate_cstr(constraints, net_out, torch_input=False):
    if len(net_out.shape) <= 1:
        net_out = net_out.reshape(1, -1)

    n_samp = net_out.shape[0]

    and_holds = (
        torch.ones(n_samp, dtype=bool, device=net_out.device)
        if torch_input
        else np.ones(n_samp, dtype=np.bool)
    )
    for or_list in constraints:
        or_holds = (
            torch.zeros(n_samp, dtype=bool, device=net_out.device)
            if torch_input
            else np.zeros(n_samp, dtype=np.bool)
        )
        for cstr in or_list:
            if cstr[0] == -1:
                or_holds = or_holds.__or__(cstr[2] > net_out[:, cstr[1]])
            elif cstr[1] == -1:
                or_holds = or_holds.__or__(net_out[:, cstr[0]] > cstr[2])
            else:
                or_holds = or_holds.__or__(
                    net_out[:, cstr[0]] - net_out[:, cstr[1]] > cstr[2]
                )
            if or_holds.all():
                break
        and_holds = and_holds.__and__(or_holds)
        if not and_holds.any():
            break
    return and_holds


def _translate_constraints_to_label(GT_specs):
    labels = []
    for and_list in GT_specs:
        label = None
        for or_list in and_list:
            if len(or_list) > 1:
                label = None
                break
            if label is None:
                label = or_list[0][0]
            elif label != or_list[0][0]:
                label = None
                break
        labels.append(label)
    return labels


def _pgd_whitebox(
    model,
    X,
    constraints,
    specLB,
    specUB,
    device,
    num_steps=50,
    step_size=0.2,
    ODI_num_steps=10,
    ODI_step_size=1.0,
    batch_size=50,
    lossFunc="margin",
    restarts=1,
    stop_early=True,
):
    out_X = model(X).detach()
    adex = None
    worst_x = None
    best_loss = torch.tensor(-np.inf)

    y = _translate_constraints_to_label([constraints])[0]

    for _ in range(restarts):
        if adex is not None:
            break
        X_pgd = Variable(
            X.data.repeat((batch_size,) + (1,) * (X.dim() - 1)), requires_grad=True
        ).to(device)
        randVector_ = torch.ones_like(model(X_pgd)).uniform_(
            -1, 1
        )  # torch.FloatTensor(*model(X_pgd).shape).uniform_(-1.,1.).to(device)
        random_noise = torch.ones_like(X_pgd).uniform_(-0.5, 0.5) * (
            specUB - specLB
        )  # torch.FloatTensor(*X_pgd.shape).uniform_(-0.5, 0.5).to(device)*(specUB-specLB)
        X_pgd = Variable(
            torch.minimum(torch.maximum(X_pgd.data + random_noise, specLB), specUB),
            requires_grad=True,
        )

        lr_scale = (specUB - specLB) / 2
        lr_scheduler = step_lr_scheduler(
            step_size,
            gamma=0.1,
            interval=[
                np.ceil(0.5 * num_steps),
                np.ceil(0.8 * num_steps),
                np.ceil(0.9 * num_steps),
            ],
        )
        gama_lambda = 10
        y_target = constraints[0][0][-1]
        regression = (
            len(constraints) == 2
            and ([(-1, 0, y_target)] in constraints)
            and ([(0, -1, y_target)] in constraints)
        )

        for i in range(ODI_num_steps + num_steps + 1):
            opt = optim.SGD([X_pgd], lr=1e-3)
            opt.zero_grad()

            with torch.enable_grad():
                out = model(X_pgd)

                cstrs_hold = _evaluate_cstr(constraints, out.detach(), torch_input=True)
                if (not regression) and (not cstrs_hold.all()) and (stop_early):
                    adv_idx = (~cstrs_hold.cpu()).nonzero(as_tuple=False)[0].item()
                    adex = X_pgd[adv_idx : adv_idx + 1]
                    assert not _evaluate_cstr(
                        constraints, model(adex), torch_input=True
                    )[0], f"{model(adex)},{constraints}"
                    # assert (specLB <= adex).all() and (specUB >= adex).all()
                    # print("Adex found via attack")
                    return [adex.detach().cpu().numpy()], None
                if i == ODI_num_steps + num_steps:
                    # print("No adex found via attack")
                    adex = None
                    break

                if i < ODI_num_steps:
                    loss = (out * randVector_).sum()
                elif lossFunc == "xent":
                    loss = nn.CrossEntropyLoss()(
                        out, torch.tensor([y] * out.shape[0], dtype=torch.long)
                    )
                elif lossFunc == "margin":
                    and_idx = np.arange(len(constraints)).repeat(
                        np.floor(batch_size / len(constraints))
                    )
                    and_idx = torch.tensor(
                        np.concatenate(
                            [and_idx, np.arange(batch_size - len(and_idx))], axis=0
                        )
                    ).to(device)
                    loss = constraint_loss(out, constraints, and_idx=and_idx).sum()
                elif lossFunc == "GAMA":
                    and_idx = np.arange(len(constraints)).repeat(
                        np.floor(batch_size / len(constraints))
                    )
                    and_idx = torch.tensor(
                        np.concatenate(
                            [and_idx, np.arange(batch_size - len(and_idx))], axis=0
                        )
                    ).to(device)
                    out = torch.softmax(out, 1)
                    loss = (
                        constraint_loss(out, constraints, and_idx=and_idx)
                        + (gama_lambda * (out_X - out) ** 2).sum(dim=1)
                    ).sum()
                    gama_lambda *= 0.9

            max_loss = torch.max(loss).item()
            if max_loss > best_loss:
                best_loss = max_loss
                worst_x = X_pgd[torch.argmax(loss)].detach().cpu().numpy()

            loss.backward()
            if i < ODI_num_steps:
                eta = ODI_step_size * lr_scale * X_pgd.grad.data.sign()
            else:
                eta = lr_scheduler.get_lr() * lr_scale * X_pgd.grad.data.sign()
                lr_scheduler.step()
            X_pgd = Variable(
                torch.minimum(torch.maximum(X_pgd.data + eta, specLB), specUB),
                requires_grad=True,
            )
    return adex, worst_x
