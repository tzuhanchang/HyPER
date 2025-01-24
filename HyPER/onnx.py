import os
import yaml
import hydra
import torch
import warnings

from omegaconf import DictConfig, OmegaConf

from HyPER.models import HyPERModel


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

    model.eval()
    with torch.no_grad():
        export_options = torch.onnx.ExportOptions(dynamic_shapes=True)
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
            cfg['onnx_output'],
            do_constant_folding=True,
            input_names=['x_s', 'edge_index', 'edge_attr_s', 'u_s', 'batch', 'edge_index_h', 'edge_index_h_batch'],
            output_names=['x_hat', 'batch_hyperedge', 'edge_attr_prime'],
            dynamic_axes={'x_s'               : {0 : 'graph_order'},
                          'edge_index'        : {1 : 'edge_size'},
                          'edge_attr_s'       : {0 : 'edge_size'},
                          'u_s'               : {0 : 'batch_size'},
                          'batch'             : {0 : 'graph_order'},
                          'edge_index_h'      : {1 : 'hyperedge_size'},
                          'edge_index_h_batch': {0 : 'hyperedge_size'},
                          'x_hat'             : {0 : 'hyperedge_size'},
                          'batch_hyperedge'   : {0 : 'hyperedge_size'},
                          'edge_attr_prime'   : {0 : 'edge_size'}}
        )

if __name__ == "__main__":
    Onnx()
