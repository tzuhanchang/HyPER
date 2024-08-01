import os
import yaml
import hydra
import torch
import warnings

from omegaconf import DictConfig, OmegaConf
from lightning_utilities.core.imports import RequirementCache
_ONNX_AVAILABLE = RequirementCache("onnx")
from typing import Any

from HyPER.models import HyPERModel


@torch.no_grad()
def to_onnx(LightningModel, output, **kwargs: Any) -> None:
    """Saves the model in ONNX format.
    This function is modified from `lightning.pytorch.core.module`.
    """
    if not _ONNX_AVAILABLE:
        raise ModuleNotFoundError(f"`{type(self).__name__}.to_onnx()` requires `onnx` to be installed.")

    mode = LightningModel.training

    torch.onnx.dynamo_export(LightningModel, **kwargs,).save(output)
    LightningModel.train(mode)


@hydra.main(version_base=None, config_path="../configs", config_name="default")
def Onnx(cfg : DictConfig) -> None:
    r"""Convert a trained network model to Onnx.

    Args:
        cfg (str): a `.yaml` file, stores training related parameters. (default: :obj:`str`=None).
    """
    print(OmegaConf.to_yaml(cfg))

    # Map location
    map_location = torch.device('cuda') if cfg['predict_with'].lower() == "gpu" else torch.device('cpu')

    # Load checkpoints
    assert cfg['convert_model'] is not None, "No model directory is provided in `convert_model`. Abort!"
    ckpt_file = [filename for filename in os.listdir(os.path.join(cfg['convert_model'], "checkpoints")) if filename.startswith("epoch")]
    if len(ckpt_file) > 1:
        warnings.warn(f"There are multiple .ckpt files listed in {cfg['convert_model']}, using the last checkpoint.")
        ckpt_file = os.path.join(cfg['convert_model'], "checkpoints", ckpt_file[-1])
    if len(ckpt_file) == 0:
        raise RuntimeError(f"No checkpoint files have been found in {cfg['convert_model']}.")
    ckpt_file = os.path.join(cfg['convert_model'], "checkpoints", ckpt_file[0])

    hparams_file = os.path.join(cfg['convert_model'], "hparams.yaml")
    assert os.path.isfile(hparams_file), f"`hparams.ymal` is not found in {cfg['convert_model']}."
    with open(hparams_file) as stream:
        hparams = yaml.safe_load(stream)

    model = HyPERModel.load_from_checkpoint(
        checkpoint_path = ckpt_file,
        hparams_file = hparams_file,
        map_location = map_location,
    )

    # Define input samples
    l_x_s_ = torch.randn((13,hparams['node_in_channels']))
    torch._dynamo.mark_dynamic(l_x_s_, 0)
    l_edge_attr_s_ = torch.randn((72,hparams['edge_in_channels']))
    torch._dynamo.mark_dynamic(l_edge_attr_s_, 0)
    l_edge_index_ = torch.randint(0,12,(2,72))
    torch._dynamo.mark_dynamic(l_edge_index_, 1)
    l_u_s_ = torch.randn((2,hparams['global_in_channels']))
    torch._dynamo.mark_dynamic(l_u_s_, 0)
    l_batch_ = torch.LongTensor([0,0,0,0,0,0,1,1,1,1,1,1,1])
    torch._dynamo.mark_dynamic(l_batch_, 0)
    l_edge_index_h_ = torch.randint(0,12,(hparams['hyperedge_order'],55))
    torch._dynamo.mark_dynamic(l_edge_index_h_, 1)
    l_edge_index_h_batch_ = torch.cat([torch.full([20],0, dtype=torch.int64),torch.full([35],1, dtype=torch.int64)],dim=0)
    torch._dynamo.mark_dynamic(l_edge_index_h_batch_, 0)

    input_samples = {
        "x_s": l_x_s_,
        "edge_attr_s": l_edge_attr_s_,
        "edge_index": l_edge_index_,
        "u_s": l_u_s_,
        "batch": l_batch_,
        "edge_index_h": l_edge_index_h_,
        "edge_index_h_batch": l_edge_index_h_batch_
    }

    to_onnx(model, output=cfg['onnx_output'], **input_samples)


if __name__ == "__main__":
    Onnx()
