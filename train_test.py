import argparse
import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch import nn
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from datasets import TrafficScopeDataset
from models import TrafficScope, TrafficScopeTemporal, TrafficScopeContextual
from metaconst import TRAFFIC_SCOPE, TRAFFIC_SCOPE_TEMPORAL, TRAFFIC_SCOPE_CONTEXTUAL
from interpretability import attention_rollout, attention_normalize
from robustness import get_robustness_test_dataset


def train_TrafficScope(data_dir, agg_scales, train_idx, batch_size,
                       temporal_seq_len, packet_len, freqs_size, agg_scale_num, agg_points_num,
                       use_temporal, use_contextual,
                       num_heads, num_layers, num_classes, dropout, learning_rate, epochs, model_path, device):
    train_dataset = TrafficScopeDataset(data_dir, agg_scales, train_idx)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    if use_temporal and use_contextual:
        model = TrafficScope(temporal_seq_len, packet_len,
                             freqs_size, agg_scale_num, agg_points_num,
                             num_heads, num_layers, num_classes, dropout)
    elif use_temporal and not use_contextual:
        model = TrafficScopeTemporal(temporal_seq_len, packet_len,
                                     num_heads, num_layers, num_classes, dropout)
    elif not use_temporal and use_contextual:
        model = TrafficScopeContextual(agg_scale_num, agg_points_num, freqs_size,
                                       num_heads, num_layers, num_classes, dropout)
    else:
        print('should specify at least one input type')
        return

    model.to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    print('load model successfully. Start training...')
    train_start_time = time.time()
    for epoch in range(epochs):
        print(f'\nEpoch {epoch+1}\n--------------------')
        epoch_start_time = time.time()
        for batch_idx, (batch_temporal_data, batch_temporal_valid_len,
                        batch_contextual_data, batch_contextual_segments, batch_labels) in enumerate(train_dataloader):
            batch_start_time = time.time()
            batch_temporal_data, batch_temporal_valid_len, \
            batch_contextual_data, batch_contextual_segments, batch_labels = \
                batch_temporal_data.to(device), batch_temporal_valid_len.to(device), \
                batch_contextual_data.to(device), batch_contextual_segments.to(device), batch_labels.to(device)
            if model.model_name == TRAFFIC_SCOPE:
                probs = model(batch_temporal_data, batch_temporal_valid_len,
                              batch_contextual_data, batch_contextual_segments)
            elif model.model_name == TRAFFIC_SCOPE_TEMPORAL:
                probs = model(batch_temporal_data, batch_temporal_valid_len)
            else:  # model.model_name = TRAFFIC_SCOPE_CONTEXTUAL
                probs = model(batch_contextual_data, batch_contextual_segments)
            loss = loss_fn(probs, batch_labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            batch_end_time = time.time()

            if batch_idx % 100 == 0:
                print(f'loss: {loss.item()}, time cost: {batch_end_time - batch_start_time} s, '
                      f'[{batch_idx + 1}]/[{len(train_dataloader)}]')
        epoch_end_time = time.time()
        print(f'Epoch time cost: {epoch_end_time - epoch_start_time} s')

    train_end_time = time.time()
    model_dir = os.path.split(model_path)[0]
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    torch.save(model, model_path)
    print(f'save {model_path} successfully')
    print(f'train {model.model_name} Done! Time cost: {train_end_time - train_start_time}')


def test_TrafficScope(data_dir, agg_scales, test_idx,
                      temporal_seq_len, packet_len, agg_scale_num, agg_points_num, freqs_size,
                      batch_size, model_path, num_classes, result_path, device,
                      robust_test_name,
                      rho=None, kappa=None, different=None, alpha=None, eta=None, beta=None, gamma=None):
    test_dataset = TrafficScopeDataset(data_dir, agg_scales, test_idx)
    if robust_test_name:
        test_dataset = get_robustness_test_dataset(test_dataset,
                                                   robust_test_name,
                                                   rho, kappa, different, alpha, eta, beta, gamma)
        print(f'generate robust test dataset with {robust_test_name} successfully')
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    data_num = len(test_dataset)

    if not os.path.exists(model_path):
        print('model path does not exist')
        return
    model = torch.load(model_path)
    model.eval()

    print('load model successfully. Start testing...')
    loss_fn = nn.CrossEntropyLoss()
    test_loss = 0

    y_preds = torch.zeros(data_num)
    y_true = torch.zeros(data_num)
    y_probs = torch.zeros((data_num, num_classes))
    # used for saving attention weights
    temporal_attention_masks = torch.zeros((data_num, temporal_seq_len, temporal_seq_len))
    contextual_attention_masks = torch.zeros((data_num, agg_scale_num*agg_points_num, agg_scale_num*agg_points_num))
    fusion_attention_masks = torch.zeros((data_num, temporal_seq_len, agg_scale_num*agg_points_num))
    # used for saving latent features
    temporal_features = torch.zeros((data_num, temporal_seq_len, packet_len))
    contextual_features = torch.zeros((data_num, agg_scale_num*agg_points_num, freqs_size))
    fusion_futures = torch.zeros((data_num, temporal_seq_len, packet_len))

    data_idx = 0
    test_start_time = time.time()
    with torch.no_grad():
        for batch_idx, (batch_temporal_data, batch_temporal_valid_len,
                        batch_contextual_data, batch_contextual_segments, batch_labels) in enumerate(test_dataloader):
            batch_temporal_data, batch_temporal_valid_len, \
            batch_contextual_data, batch_contextual_segments, batch_labels = \
                batch_temporal_data.to(device), batch_temporal_valid_len.to(device), \
                batch_contextual_data.to(device), batch_contextual_segments.to(device), batch_labels.to(device)
            if model.model_name == TRAFFIC_SCOPE:
                probs = model(batch_temporal_data, batch_temporal_valid_len,
                              batch_contextual_data, batch_contextual_segments)
                batch_temporal_attention_masks = attention_rollout(model.get_temporal_attention_weights(),
                                                                   discard_ratio=0.3)
                batch_contextual_attention_masks = attention_rollout(model.get_contextual_attention_weights(),
                                                                     discard_ratio=0.3)
                batch_fusion_attention_masks = attention_normalize(model.get_fusion_attention_weights(),
                                                                   discard_ratio=0.3)
                batch_temporal_features = model.get_temporal_features()
                batch_contextual_features = model.get_contextual_features()
                batch_fusion_features = model.get_fusion_features()
                # print(len(model.temporal_encoder.attention_weights))
                # print(model.temporal_encoder.attention_weights[0].shape)
                # print(model.contextual_encoder.attention_weights[0].shape)
            elif model.model_name == TRAFFIC_SCOPE_TEMPORAL:
                probs = model(batch_temporal_data, batch_temporal_valid_len)
            else:  # model.model_name = TRAFFIC_SCOPE_CONTEXTUAL
                probs = model(batch_contextual_data, batch_contextual_segments)

            test_loss += loss_fn(probs, batch_labels).item()
            preds = probs.argmax(1)
            batch_len = probs.size(0)
            y_preds[data_idx:data_idx + batch_len] = preds.cpu()
            y_probs[data_idx:data_idx + batch_len] = probs.cpu()
            y_true[data_idx:data_idx + batch_len] = batch_labels.cpu()

            if model.model_name == TRAFFIC_SCOPE:
                temporal_attention_masks[data_idx:data_idx+batch_len] = batch_temporal_attention_masks.cpu()
                contextual_attention_masks[data_idx:data_idx+batch_len] = batch_contextual_attention_masks.cpu()
                fusion_attention_masks[data_idx:data_idx+batch_len] = batch_fusion_attention_masks.cpu()

                temporal_features[data_idx:data_idx+batch_len] = batch_temporal_features.cpu()
                contextual_features[data_idx:data_idx+batch_len] = batch_contextual_features.cpu()
                fusion_futures[data_idx:data_idx+batch_len] = batch_fusion_features.cpu()
            data_idx += batch_len

    acc = accuracy_score(y_true.numpy(), y_preds.numpy())
    pre = precision_score(y_true.numpy(), y_preds.numpy(), average='macro')
    rec = recall_score(y_true.numpy(), y_preds.numpy(), average='macro')
    f1 = f1_score(y_true.numpy(), y_preds.numpy(), average='macro')
    print(f'\nTest loss: {test_loss / len(test_dataloader)}\n'
          f'Acc: {acc} Pre: {pre} Rec: {rec} F1: {f1}\n'
          f'time cost: {time.time() - test_start_time}')
    print(f'test {model.model_name} Done!')
    # save results
    result_dir, result_name = os.path.split(result_path)
    y_true_name = os.path.splitext(result_name)[0] + '_y_true.npy'
    y_preds_name = os.path.splitext(result_name)[0] + '_y_preds.npy'
    y_probs_name = os.path.splitext(result_name)[0] + '_y_probs.npy'
    temporal_attention_masks_name = os.path.splitext(result_name)[0] + '_temporal_attention_masks.npy'
    contextual_attention_masks_name = os.path.splitext(result_name)[0] + '_contextual_attention_masks.npy'
    fusion_attention_masks_name = os.path.splitext(result_name)[0] + '_fusion_attention_masks.npy'
    temporal_features_name = os.path.splitext(result_name)[0] + '_temporal_features.npy'
    contextual_features_name = os.path.splitext(result_name)[0] + '_contextual_features.npy'
    fusion_features_name = os.path.splitext(result_name)[0] + '_fusion_features.npy'

    y_true_path = os.path.join(result_dir, y_true_name)
    y_preds_path = os.path.join(result_dir, y_preds_name)
    y_probs_path = os.path.join(result_dir, y_probs_name)
    temporal_attention_masks_path = os.path.join(result_dir, temporal_attention_masks_name)
    contextual_attention_masks_path = os.path.join(result_dir, contextual_attention_masks_name)
    fusion_attention_masks_path = os.path.join(result_dir, fusion_attention_masks_name)
    temporal_features_path = os.path.join(result_dir, temporal_features_name)
    contextual_features_path = os.path.join(result_dir, contextual_features_name)
    fusion_features_path = os.path.join(result_dir, fusion_features_name)

    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    np.save(y_true_path, y_true.numpy())
    np.save(y_preds_path, y_preds.numpy())
    np.save(y_probs_path, y_probs.numpy())
    print(f'save {y_true_path}, {y_preds_path}, {y_probs_path} successfully')
    if model.model_name == TRAFFIC_SCOPE:
        np.save(temporal_attention_masks_path, temporal_attention_masks.numpy())
        np.save(contextual_attention_masks_path, contextual_attention_masks.numpy())
        np.save(fusion_attention_masks_path, fusion_attention_masks.numpy())
        np.save(temporal_features_path, temporal_features.numpy())
        np.save(contextual_features_path, contextual_features.numpy())
        np.save(fusion_features_path, fusion_futures.numpy())
        print(f'save {temporal_attention_masks_path}, {contextual_attention_masks_path}, '
              f'{fusion_attention_masks_path}, '
              f'{temporal_features_path}, {contextual_features_path}, {fusion_features_path} successfully')


def train_test_helper(data_dir, agg_scales, model_name, agg_scale_num, agg_points_num, batch_size,
                      temporal_seq_len, packet_len, freqs_size, use_temporal, use_contextual, is_train, is_test,
                      num_heads, num_layers, num_classes, dropout, learning_rate, epochs, model_path, result_path,
                      robust_test_name,
                      rho=None, kappa=None, different=None, alpha=None, eta=None, beta=None, gamma=None,
                      k_fold=None, gpu_id=0):
    os.environ['CUDA_VISIBLE_DEVICE'] = str(gpu_id)
    dataset = TrafficScopeDataset(data_dir, agg_scales)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if k_fold:
        skf = StratifiedKFold(n_splits=k_fold, shuffle=True, random_state=42)
        for train_idx, test_idx in skf.split(dataset.temporal_data.numpy(), dataset.labels.numpy()):
            if model_name == TRAFFIC_SCOPE:
                if is_train:
                    train_TrafficScope(data_dir, agg_scales, train_idx, batch_size,
                                       temporal_seq_len, packet_len, freqs_size, agg_scale_num, agg_points_num,
                                       use_temporal, use_contextual,
                                       num_heads, num_layers, num_classes, dropout, learning_rate, epochs, model_path,
                                       device)
                if is_test:
                    test_TrafficScope(data_dir, agg_scales, test_idx,
                                      temporal_seq_len, packet_len, agg_scale_num, agg_points_num, freqs_size,
                                      batch_size, model_path, num_classes, result_path, device,
                                      robust_test_name,
                                      rho, kappa, different, alpha, eta, beta, gamma)
    else:
        indices = np.arange(dataset.temporal_data.shape[0])
        train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42, shuffle=True)
        if model_name == TRAFFIC_SCOPE:
            if is_train:
                train_TrafficScope(data_dir, agg_scales, train_idx, batch_size,
                                   temporal_seq_len, packet_len, freqs_size, agg_scale_num, agg_points_num,
                                   use_temporal, use_contextual,
                                   num_heads, num_layers, num_classes, dropout, learning_rate, epochs, model_path,
                                   device)
            if is_test:
                test_TrafficScope(data_dir, agg_scales, test_idx,
                                  temporal_seq_len, packet_len, agg_scale_num, agg_points_num, freqs_size,
                                  batch_size, model_path, num_classes, result_path, device,
                                  robust_test_name,
                                  rho, kappa, different, alpha, eta, beta, gamma)


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--data_dir', type=str, required=True)
    args.add_argument('--agg_scales', type=str, default='[0, 1, 2]', required=False)
    args.add_argument('--model_name', type=str, default='TrafficScope', required=False)
    args.add_argument('--agg_scale_num', type=int, default=3, required=False)
    args.add_argument('--agg_points_num', type=int, default=128, required=False)
    args.add_argument('--batch_size', type=int, default=32, required=False)
    args.add_argument('--temporal_seq_len', type=int, default=64, required=False)
    args.add_argument('--packet_len', type=int, default=64, required=False)
    args.add_argument('--freqs_size', type=int, default=128, required=False)
    args.add_argument('--use_temporal', action='store_true', required=False)
    args.add_argument('--use_contextual', action='store_true', required=False)
    args.add_argument('--is_train', action='store_true', required=False)
    args.add_argument('--is_test', action='store_true', required=False)
    args.add_argument('--num_heads', type=int, default=8, required=False)
    args.add_argument('--num_layers', type=int, default=2, required=False)
    args.add_argument('--num_classes', type=int, required=True)
    args.add_argument('--dropout', type=float, default=0.5, required=False)
    args.add_argument('--lr', type=float, default=0.001, required=False)
    args.add_argument('--epochs', type=int, default=10, required=False)
    args.add_argument('--model_path', type=str, required=True)
    args.add_argument('--result_path', type=str, required=True)
    args.add_argument('--k_fold', type=int, default=None, required=False)
    args.add_argument('--gpu_id', type=int, default=0, required=False)
    args.add_argument('--robust_test_name', type=str, default=None, required=False)
    args.add_argument('--rho', type=float, default=None, required=False)
    args.add_argument('--kappa', type=int, default=None, required=False)
    args.add_argument('--different', action='store_true', default=None, required=False)
    args.add_argument('--alpha', type=float, default=None, required=False)
    args.add_argument('--eta', type=int, default=None, required=False)
    args.add_argument('--beta', type=float, default=None, required=False)
    args.add_argument('--gamma', type=float, default=None, required=False)
    args = args.parse_args()
    print(args)

    train_test_helper(args.data_dir, eval(args.agg_scales), args.model_name,
                      args.agg_scale_num, args.agg_points_num, args.batch_size,
                      args.temporal_seq_len, args.packet_len, args.freqs_size,
                      args.use_temporal, args.use_contextual, args.is_train, args.is_test,
                      args.num_heads, args.num_layers, args.num_classes,
                      args.dropout, args.lr, args.epochs, args.model_path, args.result_path,
                      args.robust_test_name,
                      args.rho, args.kappa, args.different, args.alpha, args.eta, args.beta, args.gamma,
                      args.k_fold, args.gpu_id)
