import os

import joblib
import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns
from sklearn.model_selection import train_test_split

from datasets import TrafficScopeDataset
from utils import is_matrix_similar, scale_matrix_to_image


def attention_rollout(attention_weights, discard_ratio=0.9, num_heads=8):
    # attention_weights: list of [batch_size, num_heads, query_size, key_size]
    # result: [batch_size, query_size, key_size] 得到每个query与key的重要性关系
    result = torch.eye(attention_weights[0].size(-1))
    with torch.no_grad():
        # 计算每一个layer的注意力
        for attention_weight in attention_weights:
            attention_weight = attention_weight.cpu()
            # 处理d2l返回的attention_weight是[batch_size*num_heads, query_size, key_size]
            if len(attention_weight.shape) == 3:
                attention_weight = attention_weight.view(-1, num_heads,
                                                         attention_weight.size(1), attention_weight.size(2))
            # 对每个头的注意力求平均
            attention_heads_fused = attention_weight.mean(axis=1)
            flat = attention_heads_fused.view(attention_heads_fused.size(0), -1)
            _, indices = flat.topk(int(flat.size(-1) * discard_ratio), dim=-1, largest=False)
            flat[:, indices] = 0  # 注意是view操作，对flat置0也会对attention_heads_fused产生影响

            I = torch.eye(attention_heads_fused.size(-1))
            a = (attention_heads_fused + 1.0 * I) / 2
            a = a / a.sum(dim=-1).unsqueeze(dim=-1)

            result = torch.matmul(a, result)

    result = result / result.max()
    return result


def attention_normalize(attention_weight, discard_ratio=0.9):
    with torch.no_grad():
        attention_weight = attention_weight.cpu()
        attention_heads_fused = attention_weight.mean(axis=1)
        flat = attention_heads_fused.view(attention_heads_fused.size(0), -1)
        # _, indices = flat.topk(int(flat.size(-1) * discard_ratio), dim=-1, largest=False)
        # flat[:, indices] = 0  # 注意是view操作，对flat置0也会对attention_heads_fused产生影响

        # a = (attention_heads_fused + 1.0) / 2
        a = attention_heads_fused
        a = a / (a.sum(dim=-1).unsqueeze(dim=-1) + 1)
        a = a / (a.max() + 1)
        return a


def tsne_helper(data_dir, fusion_features_path):
    data_dir = '/Users/bil369/Downloads/test/data'
    fusion_features_path = '/Users/bil369/Downloads/test/results/trafficscope_fusion_features.npy'

    dataset = TrafficScopeDataset(data_dir, [0, 1, 2])
    indices = np.arange(dataset.temporal_data.shape[0])
    train_idx, test_idx = train_test_split(indices, test_size=0.3, random_state=42, shuffle=True)
    dataset = TrafficScopeDataset(data_dir, [0, 1, 2], test_idx)
    data_num = dataset.temporal_data.shape[0]
    temporal_data = dataset.temporal_data.reshape((data_num, -1))

    fusion_features = np.load(fusion_features_path)
    fusion_features = fusion_features.reshape((data_num, -1))

    def plot_tsne(data, test_data_labels, save_name):
        labels = ['Benign', 'Bot', 'DDoS', 'DoS', 'Patator', 'PortScan']
        markers = ['.', 'p', '*', 'v', '^', '<']
        # labels = ['hdu', 'zju']
        # markers = ['.', 'p']

        fig, ax = plt.subplots()
        for i in range(len(labels)):
            class_data = data[test_data_labels == i]
            class_embedded = TSNE(n_components=2, learning_rate='auto',
                                  init='random', random_state=42).fit_transform(class_data)
            ax.scatter(class_embedded[:, 0], class_embedded[:, 1], label=labels[i], s=30, marker=markers[i])
        ax.set(xticks=[], yticks=[])
        ax.legend(ncol=3, fontsize=15)
        ax.grid(False)
        fig.tight_layout()
        plt.show()
        # plt.savefig(os.path.join('./figs', save_name), dpi=300)

    plot_tsne(temporal_data, dataset.labels, 'ids2017_temporal_data_tsne.pdf')
    plot_tsne(fusion_features, dataset.labels, 'ids2017_fusion_features_tsne.pdf')


def plot_temporal_data(temporal_data, title, save_name):
    fig, ax = plt.subplots(subplot_kw={'xticks': [], 'yticks': []})
    ax.imshow(temporal_data, interpolation='bilinear', cmap='Blues')
    ax.set_title(title)
    fig.tight_layout()
    plt.show()
    # plt.savefig(os.path.join('./figs', save_name), dpi=300)


def plot_attention_weight(attention_weight, title, save_name):
    fig, ax = plt.subplots(subplot_kw={'xticks': [], 'yticks': []})
    ax.imshow(attention_weight, interpolation='bilinear', cmap='viridis')
    ax.set_title()
    fig.tight_layout()
    plt.show()
    # plt.savefig(os.path.join('./figs', save_name), dpi=300)


def find_similar_patterns_from_attention_weights(attention_weights_path, data_dir):
    def cal_coverage(attention_weight, class_attention_weights):
        coverage = 0
        for aw in class_attention_weights:
            if is_matrix_similar(attention_weight, aw):
                coverage += 1
        return coverage / class_attention_weights.shape[0]

    def cal_distinction(attention_weight, label, attention_weights, labels):
        distinct = 0
        total = 0
        for i in range(attention_weights.shape[0]):
            if is_matrix_similar(attention_weight, attention_weights[i]):
                total += 1
                if label == labels[i]:
                    distinct += 1
        return distinct / total

    def plot_similar_patterns(class_sample_score_top10, attention_weights, save_name):
        fig, axs = plt.subplots(nrows=1, ncols=10, figsize=(50, 5),
                                subplot_kw={'xticks': [], 'yticks': []})
        for idx, ax in enumerate(axs.flat):
            indice, coverage, distinct, score = class_sample_score_top10[idx]
            ax.imshow(scale_matrix_to_image(attention_weights[indice]), cmap='viridis', interpolation='bilinear')
            ax.set_title(f'#{indice} score: {score:.2f}')
        fig.tight_layout()
        plt.show()
        # plt.savefig(os.path.join('./figs', save_name), dpi=300)

    labels_idx_to_str = ['Benign', 'Bot', 'DDoS', 'DoS', 'Patator', 'PortScan']
    # labels_idx_to_str = ['hdu', 'zju']
    attention_weights_path = '/Users/bil369/Downloads/trafficscope_ids_2017_fusion_attention_masks.npy'

    # data_dir = '/Users/bil369/Downloads/test/data/'
    # dataset = TrafficScopeDataset(data_dir, [0, 1, 2])
    # indices = np.arange(dataset.temporal_data.shape[0])
    # train_idx, test_idx = train_test_split(indices, test_size=0.3, random_state=42, shuffle=True)
    # test_dataset = TrafficScopeDataset(data_dir, [0, 1, 2], test_idx)

    attention_weights = np.load(attention_weights_path)

    test_labels = joblib.load('/Users/bil369/Downloads/ids2017_test_labels.joblib')
    class_sample_score = {}
    attention_weights = attention_weights[:1000]
    test_labels = test_labels[:1000]
    for i in range(attention_weights.shape[0]):
        print(i)
        # label = test_dataset.labels[i]
        label = test_labels[i]
        # class_attention_weights = attention_weights[test_dataset.labels == label]
        class_attention_weights = attention_weights[test_labels == label]
        coverage = cal_coverage(attention_weights[i], class_attention_weights)
        # distinct = cal_distinction(attention_weights[i], label, attention_weights, test_dataset.labels)
        distinct = cal_distinction(attention_weights[i], label, attention_weights, test_labels)
        score = 2 * coverage * distinct / (coverage + distinct)
        class_sample_score.setdefault(labels_idx_to_str[int(label)], list())
        class_sample_score[labels_idx_to_str[int(label)]].append((i, coverage, distinct, score))

    for k, v in class_sample_score.items():
        class_sample_score[k] = sorted(v, key=lambda x: x[3], reverse=True)
        print(class_sample_score[k][:10])
        plot_similar_patterns(class_sample_score[k][:10], attention_weights, f'similar_patterns_ids2017_{k}.pdf')


if __name__ == '__main__':
    # attention_mask = attention_rollout([torch.ones(2, 8, 64, 64), torch.ones(2, 8, 64, 64)])
    # print(attention_mask)
    # print(attention_mask.shape)

    # tsne_helper('', '')
    find_similar_patterns_from_attention_weights('', '')
