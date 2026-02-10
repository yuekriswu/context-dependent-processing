import torch

from niarb.nn.parameter import Parameter

__all__ = ["reset_parameters", "state_dict", "load_state_dict", "param_dict"]


def reset_parameters(module: torch.nn.Module):
    if hasattr(module, "reset_parameters"):
        module.reset_parameters()


def state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    state_dict = model.state_dict()
    for name, param in model.named_parameters():
        if isinstance(param, Parameter) and param.tag is not None and name != param.tag:
            state_dict[param.tag] = state_dict[name]
            del state_dict[name]

    return state_dict


def load_state_dict(
    model: torch.nn.Module, state_dict: dict[str, torch.Tensor], **kwargs
) -> torch.nn.Module:
    new_state_dict = {}
    for name, param in model.named_parameters():
        if (
            isinstance(param, Parameter)
            and param.tag is not None
            and param.tag in state_dict
        ):
            new_state_dict[name] = state_dict[param.tag]

    return model.load_state_dict(new_state_dict, **kwargs)


def param_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    param_dict = {}
    for name, param in model.named_parameters():
        if isinstance(param, Parameter) and param.tag is not None:
            param_dict[param.tag] = param
        else:
            param_dict[name] = param

    return param_dict
