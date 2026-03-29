import logging
import os
import random
import numpy as np
import pandas as pd
import torch

from ArgumentsSetup import assign_arguments
from Clients import client
from log import (add_file_handler, get_logger, set_formatter, set_level, stop_logger)
from process_initialization import forward
from ProcessTaskQueue import CommunicationNetwork
from SaveSharedArray import Load_data_into_shared_array_allClient
from Server import server
from torch_process_pool import TorchProcessPool
from utils import ResultsSaveDirection

# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)
random.seed(SEED)

if __name__ == "__main__":

    # Configuration Initialization
    (
        Common_config,
        Client_config_list
    ) = assign_arguments()

    # set the results saving direction
    (
        Client_base_dir_list,
        base_dir_logger,
        base_dir_temp_local_models,
        base_dir_client_weights
    ) = ResultsSaveDirection(Common_config, Client_config_list)

    # Initialize the results saving array
    results_allsubs_columns = [
        "subs",
        "test_acc_validloss",
        "test_loss_validloss",
        "test_acc_validacc",
        "test_loss_validacc",
    ]
    results_allsubs_list = []
    results_allsubs_save_path_list = []
    for idx_client in range(Common_config["num_clients"]):
        results_allsubs_list.append([])
        results_allsubs_save_path = os.path.join(Client_base_dir_list[idx_client], "Allsubs_results.csv")
        results_allsubs_save_path_list.append(results_allsubs_save_path)

    # set the logger saving direction
    add_file_handler(base_dir_logger + "/logger.txt")
    set_level(logging.INFO)
    formatter = logging.Formatter("%(asctime)s : %(message)s")
    set_formatter(formatter)

    # load the data into shared array
    Data_array_list = Load_data_into_shared_array_allClient(
        Common_config,
        Client_config_list
    )

    fold_partition_list = []
    train_folds = Common_config["train_folds"]
    for idx_client in range(Common_config["num_clients"]):
        client_config = Client_config_list[idx_client]
        num_subjects = client_config["num_subjects"]

        # 校验subject数量是否满足要求
        if num_subjects < train_folds:
            raise ValueError(
                f"Client {idx_client} has only {num_subjects} subjects, but need at least {train_folds} for 5-fold training!")

        client_partitions = []
        for i in range(train_folds):
            # 第i次训练（从0开始）：取倒数第i+1个subject为测试集（i从0到4对应倒数第1到倒数第5）
            test_subject = num_subjects - (i + 1)  # 测试集subject索引
            train_val_subjects = [s for s in range(num_subjects) if s != test_subject]  # 剩余为训练+验证集

            client_partitions.append([test_subject])
            get_logger().info(f"Client {idx_client} - Train {i + 1}/{train_folds}: total {num_subjects} subjects, "
                              f"train+val folds {train_val_subjects}, "
                              f"test fold {test_subject}")
        fold_partition_list.append(client_partitions)

    for idx_fold in range(train_folds):
        get_logger().info(
            f"============ Train {idx_fold + 1}/{train_folds} (fixed split: test on last {idx_fold + 1}th subject) ============")

        queue_network = CommunicationNetwork(Common_config)
        pool = TorchProcessPool(
            initargs=[[], {}, {"queue_network": queue_network}],
        )

        # spaw the process for the server
        pool.submit(
            forward,
            server,
            Common_config,
            Client_config_list,
            idx_fold,
            base_dir_temp_local_models,
            base_dir_client_weights
        )

        # spaw the process for the clients
        client_future_list = []
        for idx_client in range(Common_config["num_clients"]):
            # get the config & results saving folder direction for the current client
            client_config = Client_config_list[idx_client]
            Client_base_dir = Client_base_dir_list[idx_client]

            # 获取当前client的划分信息
            partition_info = fold_partition_list[idx_client]
            test_index = partition_info[idx_fold]  # 取第一个（也是唯一一个）fold

            # spaw the process for the client
            client_future = pool.submit(
                forward,
                client,
                idx_client,
                idx_fold,
                test_index,
                Client_base_dir,
                base_dir_temp_local_models,
                Common_config,
                client_config,
                Data_array_list[idx_client],
            )
            client_future_list.append(client_future)

        pool.wait_results()

        for idx_client in range(Common_config["num_clients"]):
            current_client_future = client_future_list[idx_client]
            current_client_results_allsubs = results_allsubs_list[idx_client]
            (
                current_client_best_test_acc_list,
                current_client_best_test_loss_list,
                current_client_optimal_test_acc_list,
                current_client_optimal_test_loss_list
            ) = current_client_future.result()

            # Record the results for the single fold
            partition_info = fold_partition_list[idx_client]
            test_index = partition_info[idx_fold]

            # 假设返回的列表长度与test_index中的fold数量对应
            for idx in range(len(test_index)):
                current_client_results_allsubs.append(
                    np.array(
                        [
                            test_index[idx] + 1,
                            current_client_best_test_acc_list[idx] if isinstance(current_client_best_test_acc_list,
                                                                                 list) else current_client_best_test_acc_list,
                            current_client_best_test_loss_list[idx] if isinstance(current_client_best_test_loss_list,
                                                                                  list) else current_client_best_test_loss_list,
                            current_client_optimal_test_acc_list[idx] if isinstance(current_client_optimal_test_acc_list,
                                                                                    list) else current_client_optimal_test_acc_list,
                            current_client_optimal_test_loss_list[idx] if isinstance(current_client_optimal_test_loss_list,
                                                                                     list) else current_client_optimal_test_loss_list,
                        ]
                    )
                )
        pool.shutdown()

    # Add the results into the final table
    for idx_client in range(Common_config["num_clients"]):
        results_allsubs = results_allsubs_list[idx_client]
        results_allsubs_save_path = results_allsubs_save_path_list[idx_client]

        if len(results_allsubs) > 0:
            final_results_allsubs = np.array(results_allsubs)
            final_results_allsubs = pd.DataFrame(
                final_results_allsubs, columns=results_allsubs_columns
            )

            # calculate the averaged accuracy
            col_mean = final_results_allsubs[
                ["test_acc_validloss",
                 "test_loss_validloss",
                 "test_acc_validacc",
                 "test_loss_validacc", ]
            ].mean()
            final_results_allsubs = pd.concat([final_results_allsubs, col_mean.to_frame().T], ignore_index=True)
            final_results_allsubs.to_csv(results_allsubs_save_path, index=False)

    get_logger().warning("end of experiments")
    stop_logger()