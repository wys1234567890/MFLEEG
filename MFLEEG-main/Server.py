import copy
import time
import torch
import os
from log import get_logger


def server(Common_config, client_config_list, idx_fold, temp_local_model_path, base_dir_client_weights, queue_network):
    """
    服务端的聚合操作
    The aggregation operation in the server
    Args:
        Common_config: the common config
        client_config_list: the list of all clients' config
        queue_network: the queues connecting all clients to the server
    """

    # ===================== 【FS 新增】 保存每一轮客户端特征 =====================
    client_feature_dict = {}  # key: idx_client, value: mean_feature
    # get the name of global layers from the target client (idx=0) at round 0.
    while True:
        if queue_network.client_has_data(0):
            w_glob_keys = queue_network.get_from_client(0)
            break
        time.sleep(0.5)

    client_weights_allrounds_list = []
    global_layers_filename_list = []
    for idx_round in range(Common_config["rounds"]):
        get_logger().info("Server: ready for round {} aggregation.".format(idx_round))
        client_feature_dict.clear()

        if idx_round == 0:
            for idx_client in range(Common_config["num_clients"]):
                while True:
                    if queue_network.client_has_data(idx_client):
                        recv_data = queue_network.get_from_client(idx_client)

                        if Common_config["server_aggregation"] == "FS":
                            # 客户端发的是 [temp_local_model_filename, mean_feature]
                            model_filename, mean_feature = recv_data
                            global_layers_filename_list.append(model_filename)
                            client_feature_dict[idx_client] = mean_feature  # 保存特征
                        # Once the client finish local training, get the updated global layers
                        else:
                            global_layers_filename_list.append(recv_data)
                        break
                    time.sleep(0.5)
        else:
            for idx_client in range(Common_config["num_clients"]):
                while True:
                    # Once the client finish local training, get the updated global layers
                    if queue_network.client_has_data(idx_client):
                        recv_data = queue_network.get_from_client(idx_client)

                        # ===================== FS 模式只接收特征，不清空文件 =====================
                        if Common_config["server_aggregation"] == "FS":
                            _, mean_feature = recv_data
                            client_feature_dict[idx_client] = mean_feature
                        break
                    time.sleep(0.5)

        # load the dictionary from the file
        global_layers_list = []
        for filename in global_layers_filename_list:
            temp_dict = torch.load(filename, map_location=torch.device('cpu'))
            global_layers_list.append(temp_dict)

        global_layer_model_weights, client_weights_list = server_aggregation(
            Common_config, global_layers_list, w_glob_keys, client_config_list, client_feature_dict
        )
        client_weights_allrounds_list.append(client_weights_list)


        # save the updated global models in the files and sent back to clients
        global_layer_model_weights_save_path = os.path.join(temp_local_model_path, "global_layers.pt")
        torch.save(global_layer_model_weights, global_layer_model_weights_save_path)
        for idx_client in range(Common_config["num_clients"]):
            queue_network.send_to_client(global_layer_model_weights_save_path, idx_client)


    Client_weights_allsubs_save_path = os.path.join(base_dir_client_weights, "Clients_weights_Fold{}.txt".format(idx_fold+1))
    txtfile = open(Client_weights_allsubs_save_path, 'w')
    for lines in client_weights_allrounds_list:
        txtfile.write(str(lines))
        txtfile.write("\n")
    txtfile.close()


def server_aggregation(Common_config, local_state_dicts, w_glob_keys, client_config_list, client_feature_dict):
    """
    The server does the aggregation on the global layers' weights with different strategy
    Args:
        Common_config: the common config
        client_config_list: the list of the clients' config
        local_state_dicts: the list of the local model weights
        w_glob_keys: the names of the global layers in the client
        target_valid_dataloader_inServer: the target validation set dataloader

    Returns:
        returns the updated global layers
    """
    if Common_config["server_aggregation"] == "Fedavg":
        global_layer_model_weights, client_weights_list = Fedavg(Common_config, local_state_dicts, w_glob_keys, client_config_list)

    if Common_config["server_aggregation"] == "EqualWeights":
        global_layer_model_weights, client_weights_list = EqualWeights(Common_config, local_state_dicts, w_glob_keys)

    if Common_config["server_aggregation"] == "FS":
        global_layer_model_weights, client_weights_list = FS(Common_config, local_state_dicts, w_glob_keys, client_config_list, client_feature_dict)

    return global_layer_model_weights, client_weights_list


def Fedavg(Common_config, local_state_dicts, w_glob_keys, client_config_list):
    global_layer_model_weights = None
    Total_samples = 0
    client_weights = []
    for idx_client in range(Common_config["num_clients"]):
        local_sate_dict = local_state_dicts[idx_client]
        sample_num = client_config_list[idx_client]["num_samples"]
        Total_samples += sample_num
        if global_layer_model_weights is None:
            global_layer_model_weights = {}
            for k in w_glob_keys:
                global_layer_model_weights[k] = copy.deepcopy(local_sate_dict[k].cpu()) * sample_num
        else:
            for k in w_glob_keys:
                global_layer_model_weights[k] += local_sate_dict[k].cpu() * sample_num
    for k in w_glob_keys:
        global_layer_model_weights[k] = (
                global_layer_model_weights[k] / Total_samples
        )
    client_weights_list = [weights / Total_samples for weights in client_weights]

    return global_layer_model_weights, client_weights_list

def EqualWeights(Common_config, local_state_dicts, w_glob_keys):
    global_layer_model_weights = None
    client_weights = []
    for idx_client in range(Common_config["num_clients"]):
        local_sate_dict = local_state_dicts[idx_client]
        if global_layer_model_weights is None:
            global_layer_model_weights = {}
            for k in w_glob_keys:
                global_layer_model_weights[k] = copy.deepcopy(local_sate_dict[k].cpu())
        else:
            for k in w_glob_keys:
                global_layer_model_weights[k] += local_sate_dict[k].cpu()
    for k in w_glob_keys:
        global_layer_model_weights[k] = (
                global_layer_model_weights[k] / Common_config["num_clients"]
        )
    return global_layer_model_weights, client_weights

def FS(Common_config, local_state_dicts, w_glob_keys, client_config_list, client_feature_dict):
    """
      基于全局-本地特征相似度的加权聚合
      1. 计算全局中心特征
      2. 计算每个客户端与全局中心的余弦相似度
      3. 相似度 × 样本量 = 最终聚合权重
      """
    num_clients = Common_config["num_clients"]
    eps = 0.05  # 防止权重为负/零

    # 1. 取出所有客户端特征
    feature_list = [client_feature_dict[idx] for idx in range(num_clients)]
    feature_tensor = torch.stack(feature_list)

    global_center = torch.mean(feature_tensor, dim=0)

    client_similarity = []
    for idx in range(num_clients):
        feat = feature_list[idx]
        sim = torch.cosine_similarity(feat.flatten().unsqueeze(0),   # 🔥 [200,1,7] → [1400] → [1,1400]
                global_center.flatten().unsqueeze(0)).item()
        sim = max(sim, eps)  # 下限阈值
        client_similarity.append(sim)

    # 4. 计算最终权重：样本数 × 相似度
    total_weight = 0.0
    client_weights = []
    client_nums = []

    for idx in range(num_clients):
        n = client_config_list[idx]["num_samples"]
        s = client_similarity[idx]
        w = n * s
        client_weights.append(w)
        total_weight += w

    # 归一化权重
    client_weights = [w / total_weight for w in client_weights]

    # 5. 加权聚合全局模型
    global_layer_model_weights = None
    for idx in range(num_clients):
        local_dict = local_state_dicts[idx]
        w = client_weights[idx]

        if global_layer_model_weights is None:
            global_layer_model_weights = {}
            for k in w_glob_keys:
                global_layer_model_weights[k] = copy.deepcopy(local_dict[k].cpu()) * w
        else:
            for k in w_glob_keys:
                global_layer_model_weights[k] += local_dict[k].cpu() * w

    return global_layer_model_weights, client_weights


