import os
import yaml
import hydra
import torch
import warnings

from omegaconf import DictConfig, OmegaConf
from torch.export.dynamic_shapes import Dim

from HyPER.models import HyPERModel


@hydra.main(version_base=None, config_path="../configs", config_name="ttbar_singlelep_nobtag")
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

    model.eval()
    onnx_program = torch.onnx.export(
        model,
        (
            torch.randn((13,hparams['node_in_channels'])),
            torch.randint(0,12,(2,72)),
            torch.randn((72,hparams['edge_in_channels'])),
            torch.randn((2,hparams['global_in_channels'])),
            torch.LongTensor([0,0,0,0,0,0,1,1,1,1,1,1,1]),
            torch.randint(0,12,(hparams['hyperedge_order'],55)),
            torch.cat([torch.full([20],0, dtype=torch.int64),torch.full([35],1, dtype=torch.int64)],dim=0)
        ),
        dynamo=True,
        opset_version=18,
        input_names=['x_s', 'edge_index', 'edge_attr_s', 'u_s', 'batch', 'edge_index_h', 'edge_index_h_batch'],
        dynamic_shapes={'x_s'               : {0 : Dim.DYNAMIC},
                        'edge_index'        : {1 : Dim.DYNAMIC},
                        'edge_attr_s'       : {0 : Dim.DYNAMIC},
                        'u_s'               : {0 : Dim.DYNAMIC},
                        'batch'             : {0 : Dim.DYNAMIC},
                        'edge_index_h'      : {1 : Dim.DYNAMIC},
                        'edge_index_h_batch': {0 : Dim.DYNAMIC}}
    )
    
    onnx_program.optimize()
    onnx_program.save(cfg['onnx_output'])

if __name__ == "__main__":
    Onnx()
