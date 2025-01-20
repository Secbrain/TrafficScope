import argparse
import binascii
import json
import os
import time

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pywt

from utils import get_contents_in_dir


def merge_pcaps(pcaps_dir, pcap_save_path):
    pcaps = get_contents_in_dir(pcaps_dir, ['.'], ['.pcap', '.pcapng'])
    cmd = f'mergecap -F pcap -w {pcap_save_path}'
    for p in pcaps:
        cmd += ' ' + p
    ret = os.system(cmd)
    if ret == 0:
        print(f'merge pcaps in {pcaps_dir} into {pcap_save_path} successfully')
    else:
        print(f'merge pcaps in {pcaps_dir} error')
        exit(1)


def filter_protocols_in_pcap(pcap_path):
    display_filter = "not (arp or dhcp) and (tcp or udp)"
    pcap_dir, pcap_name = os.path.split(pcap_path)
    pcap_name = os.path.splitext(pcap_name)[0]
    out_path = os.path.join(pcap_dir, pcap_name)
    cmd = f'tshark -F pcap -r {pcap_path} -w {out_path}_tmp.pcap -Y "{display_filter}"'
    ret = os.system(cmd)
    if ret == 0:
        print(f'filter protocols with display filter {display_filter} in pcap successfully')
    else:
        print(f'filter protocols in pcap error')
        exit(1)
    os.system(f'rm -f {pcap_path}')
    os.system(f'mv {out_path}_tmp.pcap {out_path}.pcap')


def split_pcap_to_sessions(pcap_path, save_dir):
    if os.path.exists(save_dir):
        os.system(f'rm -rf {save_dir}')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    ret = os.system(f'editcap -F pcap {pcap_path} - | mono SplitCap.exe -r - -s session -o {save_dir} -p 1000')
    if ret == 0:
        print(f'split {pcap_path} to sessions successfully')
    else:
        print(f'split {pcap_path} error')
        exit(1)


def parse_session_pcap_to_matrix(session_pcap_path, session_len, packet_len, packet_offset):
    """
    pcap format refer to https://www.cnblogs.com/Chary/articles/15716063.html
    :return session_matrix, padding_mask / None, None when sessioin is too short (< 3 packets)
    """
    with open(session_pcap_path, 'rb') as f:
        content = f.read()
    hexc = binascii.hexlify(content)

    # 
    if hexc[:8] == b'd4c3b2a1':
        little_endian = True
    else:
        little_endian = False

    # remove global packet header 24 bytes
    hexc = hexc[48:]

    # parse packet raw bytes
    packets_dec = []
    while len(hexc) > 0 and len(packets_dec) < session_len:
        frame_len = hexc[16:24]
        if little_endian:
            frame_len = binascii.hexlify(binascii.unhexlify(frame_len)[::-1])  # reverse str due to little endian
        frame_len = int(frame_len, 16)

        hexc = hexc[32:]  # remove current packet header 16 bytes
        frame_hex = hexc[packet_offset * 2:min(packet_len * 2, frame_len * 2)]
        frame_dec = [int(frame_hex[i:i + 2], 16) for i in range(0, len(frame_hex), 2)]
        packets_dec.append(frame_dec)

        hexc = hexc[frame_len * 2:]

    if len(packets_dec) < 3:
        return None, None

    # padding and build session matrix
    packets_dec_matrix = pd.DataFrame(packets_dec).fillna(-1).values.astype(np.uint8)
    session_matrix = np.ones((session_len, packet_len), dtype=np.uint8) * -1
    row_idx = min(packets_dec_matrix.shape[0], session_len)
    col_idx = min(packets_dec_matrix.shape[1], packet_len)
    session_matrix[:row_idx, :col_idx] = packets_dec_matrix[:row_idx, :col_idx]

    # set -1 for irrelevant features 34 35 36 37 40 41
    common_irr_fea_idx = [18, 19, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33]
    tcp_irr_fea_idx = [38, 39, 40, 41, 42, 43, 44, 45, 50, 51]
    udp_irr_fea_idx = [40, 41]
    common_irr_fea_idx = [idx - packet_offset for idx in common_irr_fea_idx]
    session_matrix[:, common_irr_fea_idx] = -1
    # 
    # TCP
    for idx in tcp_irr_fea_idx:
        session_matrix[session_matrix[:, 23 - packet_offset] == 6, idx - packet_offset] = -1
    # UDP
    for idx in udp_irr_fea_idx:
        session_matrix[session_matrix[:, 23 - packet_offset] == 17, idx - packet_offset] = -1

    return session_matrix, (session_matrix == -1).astype(np.uint8)  # return session_matrix, padding_mask


def parse_pcap_metadata(pcap_path):
    """
    :return: pcap_metadata (pd.DataFrame)
    """
    # tshark  -T fields $fields -r xxx.pcap -E header=y -E separator=, -E occurrence=f > xxx.csv
    # $fields = -e xxx -e xxx ...
    pcap_dir, pcap_name = os.path.split(pcap_path)
    csv_name = os.path.splitext(pcap_name)[0] + '.csv'
    csv_path = os.path.join(pcap_dir, csv_name)
    fields = '-e frame.time_epoch -e frame.len -e ip.src -e ip.dst -e ipv6.src -e ipv6.dst ' \
             '-e tcp.srcport -e tcp.dstport ' \
             '-e tcp.flags.urg -e tcp.flags.ack -e tcp.flags.push -e tcp.flags.reset -e tcp.flags.syn -e tcp.flags.fin ' \
             '-e udp.srcport -e udp.dstport'
    cmd = f'tshark -T fields {fields} -r {pcap_path} -E header=y -E separator=, -E occurrence=f > {csv_path}'
    ret = os.system(cmd)
    if ret == 0:
        print(f'parse {pcap_path} metadata successfully')
    else:
        print(f'parse {pcap_path} error')
        exit(1)

    pcap_metadata = pd.read_csv(csv_path)
    return pcap_metadata


def get_session_start_time(pcap_metadata, session_pcap_path):
    """
    :return: session_start_time (float), five_tuple_key (str), pcap_metadata (pd.DataFrame)
    """
    # parse five tuple info from session split file name
    _, session_pcap_name = os.path.split(session_pcap_path)
    five_tuple_info = session_pcap_name.split('.')[1]
    protocol, src_ip, src_port, dst_ip, dst_port = five_tuple_info.split('_')
    if 'a' in src_ip or 'b' in src_ip or 'c' in src_ip or 'd' in src_ip or 'e' in src_ip or 'f' in src_ip:
        src_ip = src_ip.replace('-', ':')
        dst_ip = dst_ip.replace('-', ':')
    else:
        src_ip = src_ip.replace('-', '.')
        dst_ip = dst_ip.replace('-', '.')
    five_tuple_key = '_'.join(sorted([src_ip, src_port, dst_ip, dst_port, protocol]))

    # get the not NaN five tuple info
    # if pcap_metadata has calculated the five_tuple_key, do not calculate again
    if 'five_tuple_key' not in pcap_metadata.columns:
        pcap_metadata.loc[pd.isnull(pcap_metadata['ipv6.src']), 'src_ip'] = \
            pcap_metadata.loc[pd.isnull(pcap_metadata['ipv6.src']), 'ip.src']
        pcap_metadata.loc[pd.isnull(pcap_metadata['ipv6.dst']), 'dst_ip'] = \
            pcap_metadata.loc[pd.isnull(pcap_metadata['ipv6.dst']), 'ip.dst']
        pcap_metadata.loc[pd.isnull(pcap_metadata['ip.src']), 'src_ip'] = \
            pcap_metadata.loc[pd.isnull(pcap_metadata['ip.src']), 'ipv6.src']
        pcap_metadata.loc[pd.isnull(pcap_metadata['ip.dst']), 'dst_ip'] = \
            pcap_metadata.loc[pd.isnull(pcap_metadata['ip.dst']), 'ipv6.dst']
        pcap_metadata.loc[pd.isnull(pcap_metadata['udp.srcport']), 'src_port'] = \
            pcap_metadata.loc[pd.isnull(pcap_metadata['udp.srcport']), 'tcp.srcport']
        pcap_metadata.loc[pd.isnull(pcap_metadata['udp.dstport']), 'dst_port'] = \
            pcap_metadata.loc[pd.isnull(pcap_metadata['udp.dstport']), 'tcp.dstport']
        pcap_metadata.loc[pd.isnull(pcap_metadata['tcp.srcport']), 'src_port'] = \
            pcap_metadata.loc[pd.isnull(pcap_metadata['tcp.srcport']), 'udp.srcport']
        pcap_metadata.loc[pd.isnull(pcap_metadata['tcp.dstport']), 'dst_port'] = \
            pcap_metadata.loc[pd.isnull(pcap_metadata['tcp.dstport']), 'udp.dstport']
        pcap_metadata.loc[pd.isnull(pcap_metadata['udp.srcport']), 'protocol'] = 'TCP'
        pcap_metadata.loc[pd.isnull(pcap_metadata['tcp.srcport']), 'protocol'] = 'UDP'

        # filter ip and ipv6 is NaN or tcp ports and udp ports is NaN
        pcap_metadata = pcap_metadata.loc[(pd.notnull(pcap_metadata['ip.src']) & pd.notnull(pcap_metadata['ip.dst'])) |
                                          (pd.notnull(pcap_metadata['ipv6.src']) & pd.notnull(
                                              pcap_metadata['ipv6.dst']))]
        pcap_metadata = pcap_metadata.loc[
            (pd.notnull(pcap_metadata['tcp.srcport']) & pd.notnull(pcap_metadata['tcp.dstport'])) |
            (pd.notnull(pcap_metadata['udp.srcport']) & pd.notnull(pcap_metadata['udp.dstport']))]
        pcap_metadata['five_tuple_key'] = pcap_metadata.apply(
            lambda row: '_'.join(sorted([row.src_ip, str(int(row.src_port)),
                                         row.dst_ip, str(int(row.dst_port)),
                                         row.protocol])), axis=1)

    return pcap_metadata.loc[pcap_metadata['five_tuple_key'] == five_tuple_key, 'frame.time_epoch'].min(), \
           five_tuple_key, pcap_metadata


def get_session_contextual_packet_len_seq(pcap_metadata, session_start_time,
                                          agg_scale, agg_name, agg_points_num, five_tuple_key, beta=0.5):
    """
    return agg_seq (ndarray), pcap_metadata (pd.DataFrane), session_features_seq (ndarray)， seqs(list) when agg_ms
    else return agg_seq (ndarray), pcap_metadata (pd.DataFrane), None
    """
    time_key = f'time_{agg_name}'
    # get original packet length sequence
    if time_key not in pcap_metadata.columns:
        pcap_metadata[time_key] = (pcap_metadata['frame.time_epoch'] / agg_scale).map(int)
    session_start_time = int(session_start_time / agg_scale)
    start_time = session_start_time - agg_points_num / 2 + 1
    end_time = session_start_time + agg_points_num / 2
    seq = pcap_metadata.loc[(pcap_metadata[time_key] >= start_time) & (pcap_metadata[time_key] <= end_time),
                            [time_key, 'frame.len']]
    session_seq = pcap_metadata.loc[(pcap_metadata[time_key] >= start_time) &
                                    (pcap_metadata[time_key] <= end_time) &
                                    (pcap_metadata['five_tuple_key'] == five_tuple_key),
                                    [time_key, 'frame.len']]

    # aggregate sequence
    seq = seq.groupby(time_key).sum()
    session_seq = session_seq.groupby(time_key).sum()
    agg_seq = np.zeros(agg_points_num)
    agg_session_seq = np.zeros(agg_points_num)
    for i in seq.index:
        agg_seq[int(i - start_time)] += seq.loc[i, 'frame.len']
    for i in session_seq.index:
        agg_session_seq[int(i - start_time)] += session_seq.loc[i, 'frame.len']
    max_agg_seq_packet_len = agg_seq.max()
    max_agg_session_seq_packet_len = agg_session_seq.max()
    agg_seq = agg_seq / max_agg_seq_packet_len * max_agg_session_seq_packet_len * beta
    agg_seq += agg_session_seq

    # get session feature sequence for baselines usage, only calculate once
    if agg_name == 'ms':
        session_features_seq = pcap_metadata.loc[pcap_metadata['five_tuple_key'] == five_tuple_key,
                                                 ['frame.len', 'frame.time_epoch', 'five_tuple_key',
                                                  'tcp.flags.urg', 'tcp.flags.ack', 'tcp.flags.push',
                                                  'tcp.flags.reset', 'tcp.flags.syn', 'tcp.flags.fin']]

        return agg_seq, pcap_metadata, session_features_seq.values.tolist(), seq.loc[:, 'frame.len'].tolist()
    else:
        return agg_seq, pcap_metadata, None


def wavelet_transform(seq, wave_name, agg_points_num):
    """
    :return: normalized spectrogram (ndarray: [freqs, t])
    """
    scales = np.arange(1, agg_points_num + 1)
    fc = pywt.central_frequency(wave_name)
    scales = 2 * fc * agg_points_num / scales
    cwtmatr, freqs = pywt.cwt(seq, scales, wave_name)  # cwtmatr: (freqs, t), freqs: (freqs, )
    spectrogram = np.log2((abs(cwtmatr)) ** 2 + 1)
    spectrogram = (spectrogram - np.min(spectrogram)) / (np.max(spectrogram) + 1)

    return spectrogram


def gen_temporal_data(pcap_path, sessions_dir, data_path, session_len=64, packet_len=64, packet_offset=14):
    """
    :return: session_pcaps_used
    """
    # split
    split_pcap_to_sessions(pcap_path, sessions_dir)
    # parse
    parse_start_time = time.time()
    session_pcaps = get_contents_in_dir(sessions_dir, ['.'], ['.pcap'])
    temporal_data = np.zeros((len(session_pcaps), session_len, packet_len))
    temporal_mask = np.zeros((len(session_pcaps), session_len, packet_len))
    session_pcaps_used = []
    idx = 0
    for session_pcap in session_pcaps:
        session_matrix, padding_mask = parse_session_pcap_to_matrix(session_pcap,
                                                                    session_len, packet_len, packet_offset)
        if session_matrix is None:
            print(f'{session_pcap} is too short (session len < 3)')
            continue
        temporal_data[idx, :, :] = session_matrix
        temporal_mask[idx, :, :] = padding_mask
        idx += 1
        session_pcaps_used.append(session_pcap)
        print(f'parse {session_pcap} successfully')
    # 
    temporal_data = temporal_data[:len(session_pcaps_used), :, :]
    temporal_mask = temporal_mask[:len(session_pcaps_used), :, :]
    parse_end_time = time.time()
    # save
    data_dir, data_name = os.path.split(data_path)
    data_name = os.path.splitext(data_name)[0] + '_temporal.npy'
    data_path = os.path.join(data_dir, data_name)
    mask_name = os.path.splitext(data_name)[0] + '_mask.npy'
    mask_path = os.path.join(data_dir, mask_name)
    used_name = os.path.splitext(data_name)[0] + '_session_used.json'
    used_path = os.path.join(data_dir, used_name)
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    np.save(data_path, temporal_data)
    np.save(mask_path, temporal_mask)
    with open(used_path, 'w+') as f:
        json.dump(session_pcaps_used, f)
    print(f'save {data_path}, {mask_path} and {used_path} successfully, '
          f'total {len(session_pcaps_used)} samples, '
          f'temporal feature extract time cost: {parse_end_time -parse_start_time} s, '
          f'average {(parse_end_time - parse_start_time) / len(session_pcaps_used)} s / session')
    # visualize
    # visualize_data(temporal_data, data_name)

    return session_pcaps_used


def gen_contextual_data(pcap_path, session_pcaps, wave_name, data_path, agg_seqs_path=None):
    """
        if agg_seqs_path is not None, use previous agg sequences. pcap_path, session_pcaps can ignore.
    """
    if agg_seqs_path is not None:
        agg_seqs = np.load(agg_seqs_path)
        data_num = agg_seqs.shape[0]
        contextual_data = np.zeros((data_num, 3, 128, 128))
        # 
        session_features_seqs = None
        seqs = None
        # agg_seqs shape: [N, 3, 128]
        for i in range(data_num):
            ms_spectrogram = wavelet_transform(agg_seqs[i, 0, :], wave_name, 128)
            s_spectrogram = wavelet_transform(agg_seqs[i, 1, :], wave_name, 128)
            min_spectrogram = wavelet_transform(agg_seqs[i, 2, :], wave_name, 128)

            contextual_data[i, 0, :, :] = ms_spectrogram[:, :]
            contextual_data[i, 1, :, :] = s_spectrogram[:, :]
            contextual_data[i, 2, :, :] = min_spectrogram[:, :]
        print(f'get contextual feature from previous contextual agg sequence {agg_seqs_path} successfully')
    else:
        # parse
        pcap_metadata = parse_pcap_metadata(pcap_path)
        # aggregate and transform
        process_start_time = time.time()
        contextual_data = np.zeros((len(session_pcaps), 3, 128, 128))
        agg_seqs = np.zeros((len(session_pcaps), 3, 128))
        session_features_seqs = []
        seqs = []
        for idx, session_pcap in enumerate(session_pcaps):
            session_start_time, five_tuple_key, pcap_metadata = get_session_start_time(pcap_metadata, session_pcap)
            # ms aggregate
            ms_agg_seq, pcap_metadata, session_features_seq, seq = get_session_contextual_packet_len_seq(pcap_metadata,
                                                                                                         session_start_time,
                                                                                                         0.001, 'ms', 128,
                                                                                                         five_tuple_key)
            # s aggregate
            s_agg_seq, pcap_metadata, _ = get_session_contextual_packet_len_seq(pcap_metadata, session_start_time,
                                                                                1, 's', 128, five_tuple_key)
            # minute aggregate
            min_agg_seq, pcap_metadata, _ = get_session_contextual_packet_len_seq(pcap_metadata, session_start_time,
                                                                                  60, 'min', 128, five_tuple_key)
            ms_spectrogram = wavelet_transform(ms_agg_seq, wave_name, 128)
            s_spectrogram = wavelet_transform(s_agg_seq, wave_name, 128)
            min_spectrogram = wavelet_transform(min_agg_seq, wave_name, 128)

            contextual_data[idx, 0, :, :] = ms_spectrogram[:, :]
            contextual_data[idx, 1, :, :] = s_spectrogram[:, :]
            contextual_data[idx, 2, :, :] = min_spectrogram[:, :]

            agg_seqs[idx, 0, :] = ms_agg_seq[:]
            agg_seqs[idx, 1, :] = s_agg_seq[:]
            agg_seqs[idx, 2, :] = min_agg_seq[:]

            # print(type(session_features_seq), type(session_features_seqs))
            # print(session_features_seq)
            session_features_seqs.append(session_features_seq)
            seqs.append(seq)

            print(f'get contextual data of {session_pcap} successfully')
        process_end_time = time.time()
        print(f'contextual feature extract time cost: {process_end_time - process_start_time} s, '
              f'average {(process_end_time - process_start_time) / len(session_pcaps)} s / session')
    # save
    data_dir, data_name = os.path.split(data_path)
    session_features_seqs_name = os.path.splitext(data_name)[0] + '_session_features_seqs.joblib'
    agg_seqs_name = os.path.splitext(data_name)[0] + '_agg_seqs.npy'
    seqs_name = os.path.splitext(data_name)[0] + '_seqs.npy'
    data_name = os.path.splitext(data_name)[0] + f'_{wave_name}_contextual.npy'
    session_features_seqs_path = os.path.join(data_dir, session_features_seqs_name)
    agg_seqs_path = os.path.join(data_dir, agg_seqs_name)
    seqs_path = os.path.join(data_dir, seqs_name)
    data_path = os.path.join(data_dir, data_name)
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    np.save(data_path, contextual_data)
    print(f'save {data_path} successfully')
    if session_features_seqs is not None:  # only when first time need to save
        joblib.dump(session_features_seqs, session_features_seqs_path)
        print(f'save {session_features_seqs_path} successfully')
        np.save(agg_seqs_path, agg_seqs)
        print(f'save {agg_seqs_path} successfully')
        joblib.dump(seqs, seqs_path)
        print(f'save {seqs_path} successfully')
    # visualize
    # visualize_data(contextual_data, data_name)


def visualize_data(data, data_name):
    def plot_figure(matrix, cmap, save_name):
        fig, axs = plt.subplots(nrows=3, ncols=3, figsize=(9, 9),
                                subplot_kw={'xticks': [], 'yticks': []})
        chosen_idx = np.random.choice(len(matrix), 9)
        for idx, ax in enumerate(axs.flat):
            ax.imshow(matrix[chosen_idx[idx]], cmap=cmap, interpolation='bilinear')
            ax.set_title(f'#{chosen_idx[idx]}')

        plt.tight_layout()
        plt.savefig(f'{save_name}.png')

    if len(data.shape) == 4:  # case for contextual data: N x agg_scales x freqs x t
        for idx, agg_scale in enumerate(['ms', 's', 'min']):
            plot_figure(data[:, idx, :, :], 'viridis', f'{os.path.splitext(data_name)[0]}_{agg_scale}')
    else:
        plot_figure(data, 'Blues', os.path.splitext(data_name)[0])


def gen_single_traffic_type_data(pcaps_path, class_name, sessions_dir, data_path, wave_name,
                                 agg_seqs_path=None, contextual=False):
    # merge pcaps of single class
    if os.path.isdir(pcaps_path):
        merge_pcaps(pcaps_path, f'{class_name}.pcap')
    else:
        ret = os.system(f'cp {pcaps_path} {class_name}.pcap')
        if ret == 0:
            print(f'copy {pcaps_path} to {class_name}.pcap successfully')
        else:
            print(f'copy {pcaps_path} to {class_name}.pcap error')

    filter_protocols_in_pcap(f'{class_name}.pcap')

    session_pcaps_used = gen_temporal_data(f'{class_name}.pcap', sessions_dir, data_path)
    print(f'{class_name} has {len(session_pcaps_used)} sessions')

    if not contextual:
        return

    gen_contextual_data(f'{class_name}.pcap', session_pcaps_used, wave_name, data_path, agg_seqs_path)


def gen_multi_traffic_type_data(pcaps_path, data_path, wave_name):
    pcaps = get_contents_in_dir(pcaps_path, '.', [])
    if os.path.isdir(pcaps[0]):
        for d in pcaps:
            class_name = os.path.split(d)[1]
            gen_single_traffic_type_data(d, class_name, f'{class_name}_sessions',
                                         os.path.join(data_path, f'{class_name}.npy'), wave_name)
    else:
        for p in pcaps:
            class_name = os.path.splitext(os.path.split(p)[1])[0]
            gen_single_traffic_type_data(p, class_name, f'{class_name}_sessions',
                                         os.path.join(data_path, f'{class_name}.npy'), wave_name)


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--multiple', action='store_true', required=False)
    args.add_argument('--contextual', action='store_true', required=False)
    args.add_argument('--pcaps_path', type=str, required=True)
    args.add_argument('--class_name', type=str, required=False)
    args.add_argument('--sessions_dir', type=str, required=False)
    args.add_argument('--data_path', type=str, required=True)
    args.add_argument('--wave_name', type=str, required=True)
    args.add_argument('--session_pcaps_used', type=str, required=False, default=None)
    args.add_argument('--agg_seqs_path', type=str, required=False, default=None)
    args = args.parse_args()
    print(args)

    # 
    # python dataset_gen.py --multiple --pcaps_path=/xxx/xxx/ --data_path=/path/to/save --wave_name='cgau8'
    if args.multiple:
        gen_multi_traffic_type_data(args.pcaps_path, args.data_path, args.wave_name)
    # 
    # python dataset_gen.py --contextual --pcaps_path=/xxx/traffic_type.pcap --session_pcaps_used=/xxx/xxx_temporal_session_used.json --wave_name=cgau8 --data_path=/xxx/xxx.npy
    elif args.contextual:
        if args.session_pcaps_used:
            with open(args.session_pcaps_used, 'r') as f:
                session_pcaps_used = json.load(f)
        gen_contextual_data(args.pcaps_path, session_pcaps_used, args.wave_name, args.data_path, args.agg_seqs_path)
    # 
    # python dataset_gen.py --pcaps_path=/xxx/xxx/traffic_type --class_name=xxx --sessions_dir=/path/to/sessions --data_path=/path/to/save --wave_name='cgau8'
    else:
        gen_single_traffic_type_data(args.pcaps_path, args.class_name, args.sessions_dir,
                                     args.data_path, args.wave_name)
