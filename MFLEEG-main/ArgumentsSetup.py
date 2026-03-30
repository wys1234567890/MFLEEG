import argparse
import torch


def assign_arguments():
    # The given arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--Server_aggregation",
        type=str,
        default="Fedavg",
        help="Aggregation strategy in the server",
    )
    parser.add_argument(
        "--Train_folds",
        type=int,
        default=5,
    )
    args = parser.parse_args()

    Common_config = {
        "num_clients": 6,
        "num_layers_keep": 3,
        "data_path": r"/root/autodl-tmp/Dataset/filtered_data",
        "subject_wise": False,
        "split_ratio": 0.1,
        "rounds": 200,
        "save_checkingpoint": False,
        "checkingpoint_step": 5,
    }

    All_Clients_config = [
        {
            "name": "KoreaU_MI",
            "filtering_setting": "0.3Hz_40Hz_cheby2_sos",
            "nChan": 62,
            "nTime": 1000,
            "num_subjects": 54,
            "num_samples": 21600,
            "batch_size": 256,
            "test_batch_size": 64,
            "optim_type": "adamW",
            "local_ep": 3,
            "momentum": 0.5,
            "lr": 0.01,
            "poolSize": {
                "LocalLayers": [(1, 3), (1, 3), (1, 3)],
                "GlobalLayers": (1, 3),
            },
            "localKernalSize": {
                "LocalLayers": [(1, 10), (1, 10), (1, 10)],
                "GlobalLayers": (1, 10),
            },
        },
        {
            "name": "Shin_MI",
            "filtering_setting": "0.3Hz_40Hz_cheby2_sos",
            "nChan": 30,
            "nTime": 1000,  # ✅ 已修正
            "num_subjects": 29,
            "num_samples": 1740,
            "batch_size": 256,
            "test_batch_size": 64,
            "optim_type": "adamW",
            "local_ep": 3,
            "momentum": 0.5,
            "lr": 0.005,
            "poolSize": {
                "LocalLayers": [(1, 3), (1, 3), (1, 3)],  # ✅ 全改为 (1,3)
                "GlobalLayers": (1, 3),
            },
            "localKernalSize": {
                "LocalLayers": [(1, 10), (1, 10), (1, 10)],  # ✅ 改为 (1,10) 统一
                "GlobalLayers": (1, 10),
            },
        },
        {
            "name": "Weibo2014",
            "filtering_setting": "0.3Hz_40Hz_cheby2_sos",
            "nChan": 60,
            "nTime": 800,
            "num_subjects": 10,
            "num_samples": 1740,
            "batch_size": 256,
            "test_batch_size": 64,
            "optim_type": "adamW",
            "local_ep": 5,
            "momentum": 0.5,
            "lr": 0.005,
            "poolSize": {
                "LocalLayers": [(1, 2), (1, 3), (1, 4)],
                "GlobalLayers": (1, 3),
            },
            "localKernalSize": {
                "LocalLayers": [(1, 8), (1, 8), (1, 8)],
                "GlobalLayers": (1, 10),
            },
        },
        {
            "name": "Cho2017",
            "filtering_setting": "0.3Hz_40Hz_cheby2_sos",
            "nChan": 64,
            "nTime": 1536,
            "num_subjects": 52,
            "num_samples": 9800,
            "batch_size": 256,
            "test_batch_size": 64,
            "optim_type": "adamW",
            "local_ep": 3,
            "momentum": 0.5,
            "lr": 0.01,
            "poolSize": {
                "LocalLayers": [(1, 4), (1, 3), (1, 3)],
                "GlobalLayers": (1, 3),
            },
            "localKernalSize": {
                "LocalLayers": [(1, 22), (1, 22), (1, 22)],
                "GlobalLayers": (1, 10),
            },
        },
        {
            "name": "MunichMI",
            "filtering_setting": "0.3Hz_40Hz_cheby2_sos",
            "nChan": 128,
            "nTime": 1750,
            "num_subjects": 10,
            "num_samples": 3000,
            "batch_size": 64,
            "test_batch_size": 64,
            "optim_type": "adamW",
            "local_ep": 5,
            "momentum": 0.5,
            "lr": 0.01,
            "poolSize": {
                "LocalLayers": [(1, 4), (1, 4), (1, 3)],
                "GlobalLayers": (1, 3),
            },
            "localKernalSize": {
                "LocalLayers": [(1, 10), (1, 10), (1, 10)],
                "GlobalLayers": (1, 10),
            },
        },
        {
            "name": "Murat2018",
            "filtering_setting": "0.3Hz_40Hz_cheby2_sos",
            "nChan": 22,
            "nTime": 200,
            "num_subjects": 5,
            "num_samples": 17515,
            "batch_size": 256,
            "test_batch_size": 64,
            "optim_type": "adamW",
            "local_ep": 5,
            "momentum": 0.5,
            "lr": 0.01,
            "poolSize": {
                "LocalLayers": [(1, 1), (1, 2), (1, 3)],
                "GlobalLayers": (1, 3),
            },
            "localKernalSize": {
                "LocalLayers": [(1, 6), (1, 6), (1, 6)],
                "GlobalLayers": (1, 10),
            },
        },
    ]

    Client_config_list = []
    Common_config["server_aggregation"] = args.Server_aggregation
    Common_config["train_folds"] = args.Train_folds

    # check the number of gpu
    num_gpu = torch.cuda.device_count()
    idx_gpu = 0

    for client_config in All_Clients_config:
        client_config["device"] = str(idx_gpu)
        if idx_gpu == num_gpu-1:
            idx_gpu = 0
        else:
            idx_gpu += 1
        Client_config_list.append(client_config)

    return Common_config, Client_config_list