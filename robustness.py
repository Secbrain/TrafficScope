import os

import joblib
import numpy as np
import torch
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split

from datasets import TrafficScopeDataset
from metaconst import CONTEXTUAL_MIX, PACKET_INJECTION, PACKET_LOSS, PACKET_REORDERING


def contextual_data_mix(dataset: TrafficScopeDataset, rho, kappa, different=False):
    for idx, (_, _, c, _, label) in enumerate(dataset):
        prob = np.random.rand()
        if prob < rho:
            for i in range(kappa):
                if different:
                    indices = np.arange(dataset.contextual_data_unpack.shape[0])
                    indices = indices[dataset.labels != label.item()]
                    chosen_indice = np.random.choice(indices)
                    chosen_contextual = torch.tensor(dataset.contextual_data_unpack[chosen_indice])
                    c = c + chosen_contextual / chosen_contextual.max() * c.max() * 0.5
                else:
                    indices = np.arange(dataset.contextual_data_unpack.shape[0])
                    indices = indices[dataset.labels == label.item()]
                    chosen_indice = np.random.choice(indices)
                    chosen_contextual = torch.tensor(dataset.contextual_data_unpack[chosen_indice])
                    c = c + chosen_contextual / chosen_contextual.max() * c.max() * 0.5

            dataset.contextual_data_unpack[idx, :, :] = c[:, :]

    return dataset


def dummy_packet_injection(dataset: TrafficScopeDataset, alpha, eta):
    for idx, (t, _, _, _, _) in enumerate(dataset):
        pre_packet = None
        repeat_cnt = 0
        for i in range(t.shape[0]):
            if pre_packet is not None:
                t[i, :] = pre_packet[:]
                repeat_cnt -= 1
                if repeat_cnt == 0:
                    pre_packet = None
            else:
                prob = np.random.rand()
                if prob < alpha:
                    pre_packet = t[i, :]
                    repeat_cnt = eta
        dataset.temporal_data[idx, :, :] = t[:, :]

    return dataset


def packet_loss(dataset, beta):
    for idx, (t, t_valid_len, _, _, _) in enumerate(dataset):
        for i in range(t.shape[0]):
            if i < t_valid_len:
                prob = np.random.rand()
                if prob < beta:
                    t[i:int(t_valid_len.item()-1), :] = t.clone()[i+1:int(t_valid_len.item()), :]
                    t[int(t_valid_len.item()-1), :] = -1
                    t_valid_len -= 1
        dataset.temporal_data[idx, :, :] = t[:, :]
        dataset.temporal_valid_len[idx] = t_valid_len
    return dataset


def packet_reordering(dataset, gamma):
    for idx, (t, t_valid_len, _, _, _) in enumerate(dataset):
        for i in range(t.shape[0]):
            prob = np.random.rand()
            if prob < gamma:
                indices = np.arange(t_valid_len)
                chosen1 = int(np.random.choice(indices))
                chosen2 = int(np.random.choice(indices))
                if chosen1 != chosen2:
                    t[chosen1, :], t[chosen2, :] = t.clone()[chosen2, :], t.clone()[chosen1, :]
        dataset.temporal_data[idx, :, :] = t[:, :]
    return dataset


def get_robustness_test_dataset(test_dataset, robust_test_name,
                                rho=None, kappa=None, different=None, alpha=None, eta=None, beta=None, gamma=None):
    if robust_test_name == CONTEXTUAL_MIX:
        test_dataset_change = contextual_data_mix(test_dataset, rho, kappa, different)
    elif robust_test_name == PACKET_INJECTION:
        test_dataset_change = dummy_packet_injection(test_dataset, alpha, eta)
    elif robust_test_name == PACKET_LOSS:
        test_dataset_change = packet_loss(test_dataset, beta)
    else:
        test_dataset_change = packet_reordering(test_dataset, gamma)

    return test_dataset_change


def plot_robustness_interpretation(data_dir, fusion_features_ori_path, fusion_features_change_path,
                                   robust_test_name,
                                   rho=None, kappa=None, different=None, alpha=None, eta=None, beta=None, gamma=None):
    # dataset = TrafficScopeDataset(data_dir, [0, 1, 2])
    # indices = np.arange(dataset.temporal_data.shape[0])
    # train_idx, test_idx = train_test_split(indices, test_size=0.3, random_state=42, shuffle=True)
    # joblib.dump(test_idx, 'test_idx.joblib')

    # test_idx = joblib.load('test_idx.joblib')
    # test_dataset_ori = TrafficScopeDataset(data_dir, [0, 1, 2], test_idx)
    # test_data_num = test_dataset_ori.temporal_data.shape[0]
    # test_temporal_data_ori = test_dataset_ori.temporal_data.reshape((test_data_num, -1))
    #
    # test_dataset_change = TrafficScopeDataset(data_dir, [0, 1, 2], test_idx)
    # test_dataset_change = get_robustness_test_dataset(test_dataset_change,
    #                                                   robust_test_name,
    #                                                   rho, kappa, different, alpha, eta, beta, gamma)
    # test_temporal_data_change = test_dataset_change.temporal_data.reshape((test_data_num, -1))
    #
    # joblib.dump(test_temporal_data_ori, 'test_temporal_data_ori_ids2017.joblib')
    # joblib.dump(test_temporal_data_change, 'test_temporal_data_change_ids2017.joblib')
    #
    # fusion_features_ori = np.load(fusion_features_ori_path)
    # fusion_features_ori = fusion_features_ori.reshape((test_data_num, -1))
    # fusion_features_change = np.load(fusion_features_change_path)
    # fusion_features_change = fusion_features_change.reshape((test_data_num, -1))
    #
    # joblib.dump(fusion_features_ori, 'fusion_features_ori_ids2017.joblib')
    # joblib.dump(fusion_features_change, 'fusion_features_change_ids2017.joblib')

    def plot_kde(data_ori, data_change, label_ori, label_change, save_name):
        ori_embedded = TSNE(n_components=1, learning_rate='auto',
                            init='random', random_state=42).fit_transform(data_ori)
        change_embedded = TSNE(n_components=1, learning_rate='auto',
                               init='random', random_state=42).fit_transform(data_change)
        plt.figure()
        sns.kdeplot(ori_embedded[:, 0], label=label_ori, cumulative=True, fill=True, common_grid=True)
        sns.kdeplot(change_embedded[:, 0], label=label_change, cumulative=True, fill=True, common_grid=True)
        plt.xlabel('Feature', fontsize=15)
        plt.ylabel('Density', fontsize=15)
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        plt.grid()
        plt.legend(loc='lower right', fontsize=15)
        plt.tight_layout()
        plt.show()
        # plt.savefig(os.path.join('./figs', save_name), dpi=300)

    test_temporal_data_ori = joblib.load('test_temporal_data_ori_ids2017.joblib')
    test_temporal_data_change = joblib.load('test_temporal_data_change_ids2017.joblib')
    # fusion_features_ori = joblib.load('fusion_features_ori_ids2017.joblib')
    # fusion_features_change = joblib.load('fusion_features_change_ids2017.joblib')
    plot_kde(test_temporal_data_ori, test_temporal_data_change, 'Original', 'Modified',
             f'ids2017_temporal_data_{robust_test_name}_kde.pdf')
    # plot_kde(fusion_features_ori, fusion_features_change, 'Original', 'Modified',
    #          f'ids2017_fusion_features_{robust_test_name}_kde.pdf')


if __name__ == '__main__':
    # data_dir = '/Users/bil369/data'
    # dataset_ori = TrafficScopeDataset(data_dir, [0, 1, 2])

    # dataset = TrafficScopeDataset(data_dir, [0, 1, 2])
    # contextual_data_mix(dataset, 0.5, 3, False)

    # dataset = TrafficScopeDataset(data_dir, [0, 1, 2])
    # dummy_packet_injection(dataset, 0.5, 5)

    # dataset = TrafficScopeDataset(data_dir, [0, 1, 2])
    # packet_loss(dataset, 0.5)

    # dataset = TrafficScopeDataset(data_dir, [0, 1, 2])
    # packet_reordering(dataset, 0.5)

    plot_robustness_interpretation('/data/IDS2017',
                                   '/TrafficScope/results/trafficscope_ids2017_fusion_features.npy',
                                   '/TrafficScope/results/trafficscope_ids2017_robust_mix_fusion_features.npy',
                                   'mix', rho=0.5, kappa=3, different=True)
