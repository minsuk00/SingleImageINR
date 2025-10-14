import tinycudann as tcnn


def build_model(cfg, device):
    return tcnn.NetworkWithInputEncoding(
        n_input_dims=2,
        n_output_dims=1,
        encoding_config=cfg["hash"]["encoding_config"],
        network_config=cfg["hash"]["network_config"],
    ).to(device)
