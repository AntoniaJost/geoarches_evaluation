import hydra
from evaltools.module import GeoClimate
from pprint import pprint as pp


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg):
    """
    Main function to run the climate evaluation.

    :param cfg: Configuration object from Hydra.
    :return: None
    """

    # Load the data

    evaluator = GeoClimate(data=cfg["data"], metric_cfgs=cfg["eval"], output_path=cfg["output_path"])
    evaluator.evaluate(cfg["target_metrics"] if "target_metrics" in cfg else None)


if __name__ == "__main__":
    main()
