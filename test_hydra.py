from omegaconf import DictConfig, OmegaConf
from hydra import initialize, compose, initialize_config_module
import hydra

def main(cfg: DictConfig) -> None:
    # cfg = compose(config_name="config")  # Načtěte konfigurační soubor config.yaml
    print(OmegaConf.to_yaml(cfg))  # Vypište konfiguraci

    print(cfg.video.source)

@hydra.main(config_path="config", config_name="config")
def hydra_main(cfg: DictConfig) -> None:
    main(cfg)

if __name__ == "__main__":
    hydra_main()