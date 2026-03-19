import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import os
import logging
import time
import math
import numpy as np
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import CosineAnnealingLR
from Bio import SeqIO
from concurrent.futures import ThreadPoolExecutor
from collections import OrderedDict

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 设置内存优化选项
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
torch.cuda.empty_cache()

# 设置全局随机种子以确保结果可重复
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
set_seed(47003)

# 词汇表（独热编码）
vocab = {'A': [1, 0, 0, 0], 'G': [0, 1, 0, 0], 'C': [0, 0, 1, 0], 'T': [0, 0, 0, 1], '[PAD]': [0, 0, 0, 0]}
input_dim = 4
vocab_map = {'A': 0, 'G': 1, 'C': 2, 'T': 3, '[PAD]': 4}
vocab_array = np.array(list(vocab.values()), dtype=np.float32)

# 批量将字符串序列转换为张量表示（向量化）
def batch_str_to_tensor(seq_strs, vocab_map, vocab_array, n_threshold=0.025):
    results = []
    valid_bases = ['A', 'G', 'C', 'T']
    ambiguous_base_map = {
        'W': ['A', 'T'], 'R': ['A', 'G'], 'Y': ['C', 'T'], 'S': ['C', 'G'],
        'M': ['A', 'C'], 'K': ['G', 'T'], 'N': ['A', 'G', 'C', 'T'],
        'B': ['C', 'G', 'T'], 'D': ['A', 'G', 'T'], 'H': ['A', 'C', 'T'],
        'V': ['A', 'C', 'G']
    }
    non_agct_chars = {}
    non_agct_count = 0
    
    for seq_idx, seq_str in enumerate(seq_strs):
        seq_str = seq_str.strip().replace(' ', '').replace('\n', '').replace('\r', '').upper()
        total_length = len(seq_str)
        n_count = seq_str.count('N')
        n_ratio = n_count / total_length if total_length > 0 else 0

        if n_ratio > n_threshold:
            logger.warning(f"Sequence {seq_idx} skipped due to high N ratio: {n_ratio:.4f}")
            results.append(None)
            continue

        seq_non_agct = {}
        for base in seq_str:
            if base not in valid_bases:
                seq_non_agct[base] = seq_non_agct.get(base, 0) + 1
                non_agct_chars[base] = non_agct_chars.get(base, 0) + 1
                non_agct_count += 1

        if seq_non_agct:
            logger.info(f"Sequence {seq_idx} contains non-AGCT characters: {seq_non_agct}")

        seq_list = list(seq_str)
        random.seed(47003)
        for i, base in enumerate(seq_list):
            if base not in valid_bases:
                possible_bases = ambiguous_base_map.get(base, ['A', 'G', 'C', 'T'])
                seq_list[i] = random.choice(possible_bases)
        random.seed(None)
        seq_str = ''.join(seq_list)
        
        seq_array = np.array([vocab_map.get(base, vocab_map['[PAD]']) for base in seq_str], dtype=np.int32)
        one_hot = vocab_array[seq_array]
        results.append(torch.tensor(one_hot, dtype=torch.float))

    if non_agct_chars:
        logger.info(f"Total non-AGCT characters found: {non_agct_count}")
        logger.info(f"Non-AGCT characters and their counts: {non_agct_chars}")
    else:
        logger.info("No non-AGCT characters found in the sequences.")

    return results

# 预处理并保存FASTA文件中的序列为张量（多线程+向量化）
def save_preprocessed_sequences(fasta_file, vocab_map, vocab_array, n_threshold=0.025, num_threads=64, batch_size=1000):
    output_file = os.path.splitext(fasta_file)[0] + "_preprocessed.pt"
    sequences = []

    try:
        with open(fasta_file, 'r') as handle:
            records = list(SeqIO.parse(handle, "fasta"))
        logger.info(f"Read {len(records)} sequences from {fasta_file}")

        def process_batch(batch_records):
            seq_strs = [str(record.seq) for record in batch_records]
            tensors = batch_str_to_tensor(seq_strs, vocab_map, vocab_array, n_threshold)
            return [t for t in tensors if t is not None and t.size(0) > 0]

        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            batches = [records[i:i + batch_size] for i in range(0, len(records), batch_size)]
            results = list(executor.map(process_batch, batches))
        
        for batch_tensors in results:
            sequences.extend(batch_tensors)

    except Exception as e:
        logger.error(f"Error reading {fasta_file}: {str(e)}")
        raise ValueError(f"Unable to read sequences from {fasta_file}")

    if not sequences:
        raise ValueError(f"No valid sequences found in {fasta_file}")

    torch.save(sequences, output_file)
    logger.info(f"Saved {len(sequences)} preprocessed sequences to {output_file}")
    return sequences

# 加载预处理后的序列张量
def load_preprocessed_sequences(preprocessed_file):
    sequences = torch.load(preprocessed_file)
    logger.info(f"Loaded preprocessed sequences from {preprocessed_file}")
    return sequences

# 预处理函数，只处理训练集
def preprocess_sequences(active_fasta, dormant_fasta, vocab_map, vocab_array, num_threads=64):
    # 处理训练集
    active_preprocessed_file = os.path.splitext(active_fasta)[0] + "_preprocessed.pt"
    if not os.path.exists(active_preprocessed_file):
        active_sequences = save_preprocessed_sequences(active_fasta, vocab_map, vocab_array, num_threads=num_threads)
    else:
        active_sequences = load_preprocessed_sequences(active_preprocessed_file)
    
    dormant_preprocessed_file = os.path.splitext(dormant_fasta)[0] + "_preprocessed.pt"
    if not os.path.exists(dormant_preprocessed_file):
        dormant_sequences = save_preprocessed_sequences(dormant_fasta, vocab_map, vocab_array, num_threads=num_threads)
    else:
        dormant_sequences = load_preprocessed_sequences(dormant_preprocessed_file)
    
    # 读取FASTA记录用于后续输出
    active_records = list(SeqIO.parse(active_fasta, "fasta"))
    dormant_records = list(SeqIO.parse(dormant_fasta, "fasta"))
    
    # 合并训练集
    train_sequences = active_sequences + dormant_sequences
    train_labels = [1] * len(active_sequences) + [0] * len(dormant_sequences)
    train_records = active_records + dormant_records
    
    # 统计序列长度信息
    active_lengths = [seq.size(0) for seq in active_sequences]
    dormant_lengths = [seq.size(0) for seq in dormant_sequences]
    
    logger.info(f"Training Active sequences: Total {len(active_sequences)}, Min length: {min(active_lengths)}, Max length: {max(active_lengths)}, Avg length: {sum(active_lengths)/len(active_lengths):.2f}")
    logger.info(f"Training Dormant sequences: Total {len(dormant_sequences)}, Min length: {min(dormant_lengths)}, Max length: {max(dormant_lengths)}, Avg length: {sum(dormant_lengths)/len(dormant_sequences):.2f}")
    logger.info(f"Training set: {len(train_sequences)} sequences ({sum(train_labels)} active, {len(train_labels) - sum(train_labels)} dormant)")
    
    return train_sequences, train_labels, train_records

# 定义prophage数据集类
class ProphageDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = sequences
        self.labels = labels

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]

# 定义孪生数据集类（修改为对比损失）
class SiameseDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = sequences
        self.labels = np.array(labels)
        self.active_indices = np.where(self.labels == 1)[0]
        self.dormant_indices = np.where(self.labels == 0)[0]
        logger.info(f"Initialized SiameseDataset: active={len(self.active_indices)} dormant={len(self.dormant_indices)}")

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        anchor_seq = self.sequences[idx]
        anchor_label = self.labels[idx]
        
        if anchor_label == 1:
            if random.random() < 0.5:
                pair_idx = random.choice(self.active_indices[self.active_indices != idx])
                pair_label = 1
            else:
                pair_idx = random.choice(self.dormant_indices)
                pair_label = 0
        else:
            pair_idx = random.choice(self.active_indices)
            pair_label = 0
        
        anchor_length = anchor_seq.size(0)
        pair_length = self.sequences[pair_idx].size(0)
        
        return (anchor_seq, torch.tensor(anchor_label, dtype=torch.long),
                self.sequences[pair_idx], torch.tensor(pair_label, dtype=torch.long),
                anchor_length, pair_length)

# 处理监督数据批次，进行填充
def supervised_collate_fn(batch):
    sequences, labels = zip(*batch)
    padded_sequences = pad_sequence(sequences, batch_first=True, padding_value=0)
    original_lengths = torch.tensor([len(seq) for seq in sequences])
    labels = torch.tensor(labels, dtype=torch.long)
    return padded_sequences, original_lengths, labels

# 处理孪生数据批次（修改为对比损失）
def siamese_collate_fn(batch):
    anchor_seqs, anchor_labels, pair_seqs, pair_labels, anchor_lengths, pair_lengths = zip(*batch)
    padded_anchor_seqs = pad_sequence(anchor_seqs, batch_first=True, padding_value=0)
    padded_pair_seqs = pad_sequence(pair_seqs, batch_first=True, padding_value=0)
    anchor_lengths = torch.tensor(anchor_lengths)
    pair_lengths = torch.tensor(pair_lengths)
    anchor_labels = torch.tensor(anchor_labels, dtype=torch.long)
    pair_labels = torch.tensor(pair_labels, dtype=torch.long)
    return padded_anchor_seqs, padded_pair_seqs, anchor_lengths, pair_lengths, anchor_labels, pair_labels

# Focal Loss实现，alpha控制正类（active，标签1）
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.75, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        log_pt = F.log_softmax(inputs, dim=-1)
        pt = torch.exp(log_pt)
        targets = targets.view(-1, 1)
        log_pt = log_pt.gather(1, targets).view(-1)
        pt = pt.gather(1, targets).view(-1)
        ce_loss = -log_pt
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss

# 对比损失（使用余弦相似度）
class ContrastiveLoss(nn.Module):
    def __init__(self, margin=0.3):
        super().__init__()
        self.margin = margin

    def forward(self, anchor_features, pair_features, pair_labels):
        anchor_features = F.normalize(anchor_features, p=2, dim=1)
        pair_features = F.normalize(pair_features, p=2, dim=1)
        
        cosine_sim = torch.sum(anchor_features * pair_features, dim=1)
        cosine_dist = 1.0 - cosine_sim
        
        positive_loss = pair_labels * cosine_dist
        negative_loss = (1 - pair_labels) * torch.relu(self.margin - cosine_dist)
        loss = (positive_loss + negative_loss).mean()
        
        if torch.isnan(loss) or torch.isinf(loss):
            logger.warning("NaN or Inf detected in Contrastive Loss, returning 0")
            return torch.tensor(0.0, device=anchor_features.device, requires_grad=True)
        
        return loss

# Warmup学习率调度器
class WarmupScheduler:
    def __init__(self, optimizer, warmup_steps, base_lrs, eta_mins, total_steps):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.base_lrs = base_lrs
        self.eta_mins = eta_mins
        self.current_step = 0
        self.total_steps = total_steps
        self.cosine_scheduler = CosineAnnealingLR(optimizer, T_max=total_steps - warmup_steps, eta_min=eta_mins[0])
        for i, param_group in enumerate(self.optimizer.param_groups):
            param_group['lr'] = self.eta_mins[i]

    def step(self, stage='pretrain'):
        self.current_step += 1
        if stage == 'joint':
            lr_scales = [2.0, 1.5, 2.0, 0.0]
        elif stage == 'finetune':
            lr_scales = [0.5, 0.5, 0.0, 1.0]
        else:
            lr_scales = [2.0, 1.0, 1.5, 1.0]

        if self.current_step <= self.warmup_steps:
            for i, param_group in enumerate(self.optimizer.param_groups):
                lr = self.eta_mins[i] + (self.base_lrs[i] * lr_scales[i] - self.eta_mins[i]) * (self.current_step / self.warmup_steps)
                param_group['lr'] = max(lr, self.eta_mins[i])
        else:
            for i, param_group in enumerate(self.optimizer.param_groups):
                param_group['lr'] = self.base_lrs[i] * lr_scales[i]
            self.cosine_scheduler.step()

        return self.current_step > self.warmup_steps

# 动态旋转位置编码(ROPE)
class RotaryPositionalEncoding(nn.Module):
    def __init__(self, d_k, max_seq_len=7000):
        super().__init__()
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        self.theta_segments = [
            (500, 10000.0), (1000, 8000.0), (1500, 6000.0), (2000, 5000.0), (None, 4500.0)
        ]
        for threshold, base_theta in self.theta_segments:
            theta = base_theta ** (-2.0 * (torch.arange(0, d_k, 2).float()) / d_k)
            self.register_buffer(f'theta_{threshold if threshold is not None else "max"}', theta)

    def get_theta_for_length(self, seq_len):
        for threshold, _ in self.theta_segments:
            if threshold is None or seq_len <= threshold:
                key = f'theta_{threshold if threshold is not None else "max"}'
                return getattr(self, key)
        return getattr(self, f'theta_max')

    def forward(self, x):
        batch_size, nhead, seq_len, d_k = x.size()
        assert d_k == self.d_k, f"Input dimension {d_k} does not match initialized d_k {self.d_k}"
        theta = self.get_theta_for_length(seq_len)
        positions = torch.arange(seq_len, device=x.device).float().unsqueeze(1)
        angles = positions * theta
        sin_vals = torch.sin(angles).unsqueeze(0).unsqueeze(1)
        cos_vals = torch.cos(angles).unsqueeze(0).unsqueeze(1)
        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]
        x_rot_even = x_even * cos_vals - x_odd * sin_vals
        x_rot_odd = x_even * sin_vals + x_odd * cos_vals
        x_rot = torch.zeros_like(x)
        x_rot[..., 0::2] = x_rot_even
        x_rot[..., 1::2] = x_rot_odd
        return x_rot

# 多头注意力机制
class RotaryMultiheadAttention(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.3, device='cpu'):
        super().__init__()
        assert d_model % nhead == 0
        self.d_model = d_model
        self.nhead = nhead
        self.d_k = d_model // nhead
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.rope = RotaryPositionalEncoding(self.d_k)
        self.initialize_weights()
        self.attn = None

    def initialize_weights(self):
        nn.init.kaiming_uniform_(self.q_proj.weight, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_uniform_(self.k_proj.weight, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_uniform_(self.v_proj.weight, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_uniform_(self.out_proj.weight, mode='fan_in', nonlinearity='relu')
        if self.q_proj.bias is not None:
            nn.init.zeros_(self.q_proj.bias)
            nn.init.zeros_(self.k_proj.bias)
            nn.init.zeros_(self.v_proj.bias)
            nn.init.zeros_(self.out_proj.bias)

    def forward(self, query, key, value, attn_mask=None, training=True):
        batch_size = query.size(0)
        seq_len = query.size(1)
        q = self.q_proj(query).view(batch_size, -1, self.nhead, self.d_k).transpose(1, 2)
        k = self.k_proj(key).view(batch_size, -1, self.nhead, self.d_k).transpose(1, 2)
        v = self.v_proj(value).view(batch_size, -1, self.nhead, self.d_k).transpose(1, 2)
        q = self.rope(q)
        k = self.rope(k)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if attn_mask is not None:
            attn_mask = attn_mask.expand(batch_size, self.nhead, seq_len, seq_len)
            scores = scores + attn_mask
        attn = torch.softmax(scores, dim=-1)
        if torch.isnan(attn).any() or torch.isinf(attn).any():
            attn = torch.zeros_like(attn)
        self.attn = attn
        attn = self.dropout(attn) if training else attn
        context = torch.matmul(attn, v)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.out_proj(context)
        return output, attn

# Transformer编码器层
class RotaryTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.3, device='cpu'):
        super().__init__()
        self.self_attn = RotaryMultiheadAttention(d_model, nhead, dropout=dropout, device=device)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = nn.GELU()
        self.initialize_weights()

    def initialize_weights(self):
        nn.init.kaiming_uniform_(self.linear1.weight, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_uniform_(self.linear2.weight, mode='fan_in', nonlinearity='relu')
        if self.linear1.bias is not None:
            nn.init.zeros_(self.linear1.bias)
            nn.init.zeros_(self.linear2.bias)

    def forward(self, src, src_mask=None, training=True):
        src2, attn = self.self_attn(src, src, src, attn_mask=src_mask, training=training)
        src = src + self.dropout1(src2) if training else src + src2
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))) if training else self.activation(self.linear1(src)))
        src = src + self.dropout2(src2) if training else src + src2
        return self.norm2(src), attn

# CNN层
class CNNLayer(nn.Module):
    def __init__(self, input_dim=4, d_model=256, window_size=256, overlap_percent=0.25, dropout=0.5):
        super().__init__()
        self.proj = nn.Linear(input_dim, d_model)
        try:
            overlap_percent = float(overlap_percent)
            if not 0 <= overlap_percent < 1:
                raise ValueError("overlap_percent must be between 0 and 1")
        except (ValueError, TypeError) as e:
            raise ValueError(f"Invalid overlap_percent: {overlap_percent}")
        
        self.window_size = window_size
        self.stride = int(window_size * (1 - overlap_percent))
        self.conv3 = nn.Sequential(
            nn.Conv1d(d_model, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(64),
        )
        self.conv5 = nn.Sequential(
            nn.Conv1d(d_model, 128, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.BatchNorm1d(128),
        )
        self.conv7 = nn.Sequential(
            nn.Conv1d(d_model, 128, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.BatchNorm1d(128),
        )
        self.conv9 = nn.Sequential(
            nn.Conv1d(d_model, 192, kernel_size=9, padding=4),
            nn.ReLU(),
            nn.BatchNorm1d(192),
        )
        self.res_proj3 = nn.Conv1d(d_model, 64, kernel_size=1)
        self.res_proj5 = nn.Conv1d(d_model, 128, kernel_size=1)
        self.res_proj7 = nn.Conv1d(d_model, 128, kernel_size=1)
        self.res_proj9 = nn.Conv1d(d_model, 192, kernel_size=1)
        self.pool = nn.MaxPool1d(kernel_size=3, stride=1)
        self.final_pool = nn.AdaptiveMaxPool1d(1)
        self.dropout = nn.Dropout(dropout)
        self.initialize_weights()

    def initialize_weights(self):
        nn.init.kaiming_uniform_(self.proj.weight, mode='fan_in', nonlinearity='relu')
        for conv in [self.conv3[0], self.conv5[0], self.conv7[0], self.conv9[0]]:
            nn.init.kaiming_uniform_(conv.weight, mode='fan_in', nonlinearity='relu')
            if conv.bias is not None:
                nn.init.zeros_(conv.bias)
        for res_proj in [self.res_proj3, self.res_proj5, self.res_proj7, self.res_proj9]:
            nn.init.kaiming_uniform_(res_proj.weight, mode='fan_in', nonlinearity='relu')
            if res_proj.bias is not None:
                nn.init.zeros_(res_proj.bias)

    def forward(self, x, original_lengths, training=True):
        batch_size, seq_len, _ = x.size()
        x = self.proj(x)
        num_windows = math.ceil((seq_len - self.window_size) / self.stride) + 1
        padded_len = (num_windows - 1) * self.stride + self.window_size
        mask = torch.ones(batch_size, num_windows, dtype=torch.bool, device=x.device)
        for i, length in enumerate(original_lengths):
            valid_windows = math.ceil((length.item() - self.window_size) / self.stride) + 1
            if valid_windows < num_windows:
                mask[i, valid_windows:] = False
        if seq_len < padded_len:
            padding = torch.zeros(batch_size, padded_len - seq_len, x.size(-1), device=x.device)
            x = torch.cat([x, padding], dim=1)
        x = x.unfold(1, self.window_size, self.stride)
        x = x.reshape(-1, x.size(2), x.size(3))
        residual = x
        x3 = self.pool(self.conv3(x) + self.res_proj3(residual))
        x5 = self.pool(self.conv5(x) + self.res_proj5(residual))
        x7 = self.pool(self.conv7(x) + self.res_proj7(residual))
        x9 = self.pool(self.conv9(x) + self.res_proj9(residual))
        combined = torch.cat([x3, x5, x7, x9], dim=1)
        
        del x3, x5, x7, x9
        torch.cuda.empty_cache()

        pooled = self.final_pool(combined)

        del combined
        x = pooled.view(batch_size, num_windows, -1)
        x = self.dropout(x) if training else x
        return x, mask

# 全连接层
class FullyConnectedLayer(nn.Module):
    def __init__(self, d_model, dropout=0.2):
        super().__init__()
        self.fc1 = nn.Linear(d_model, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 2)
        self.residual_proj1 = nn.Linear(d_model, 512)
        self.residual_proj2 = nn.Linear(512, 256)
        self.norm1 = nn.LayerNorm(512)
        self.norm2 = nn.LayerNorm(256)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.initialize_weights()

    def initialize_weights(self):
        nn.init.kaiming_uniform_(self.fc1.weight, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_uniform_(self.fc2.weight, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_uniform_(self.fc3.weight, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_uniform_(self.residual_proj1.weight, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_uniform_(self.residual_proj2.weight, mode='fan_in', nonlinearity='relu')
        if self.fc1.bias is not None:
            nn.init.zeros_(self.fc1.bias)
            nn.init.zeros_(self.fc2.bias)
            nn.init.zeros_(self.fc3.bias)
            nn.init.zeros_(self.residual_proj1.bias)
            nn.init.zeros_(self.residual_proj2.bias)

    def forward(self, x):
        res1 = self.residual_proj1(x)
        x = self.gelu(self.fc1(x))
        x = self.dropout(x)
        x = self.norm1(x + res1)
        res2 = self.residual_proj2(x)
        x = self.gelu(self.fc2(x))
        x = self.dropout(x)
        x = self.norm2(x + res2)
        x = self.fc3(x)
        return x

# Transformer层
class TransformerLayer(nn.Module):
    def __init__(self, d_model=512, nhead=16, num_layers=2, dropout=0.3, cls_dropout=0.1, device='cpu'):
        super().__init__()
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.transformer_layers = nn.ModuleList([
            RotaryTransformerEncoderLayer(d_model, nhead, d_model*4, dropout, device)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.nhead = nhead

    def create_padding_mask(self, mask, device):
        batch_size, num_windows = mask.size()
        seq_len = num_windows + 1
        cls_mask = torch.ones(batch_size, 1, dtype=torch.bool, device=device)
        extended_mask = torch.cat([cls_mask, mask], dim=1)
        attn_mask = torch.zeros(batch_size, self.nhead, seq_len, seq_len, device=device)
        invalid_mask = ~extended_mask.unsqueeze(-1) * ~extended_mask.unsqueeze(-2)
        attn_mask.masked_fill_(invalid_mask.unsqueeze(1), float('-inf'))
        return attn_mask

    def forward(self, x, mask, original_lengths, training=True):
        batch_size = x.size(0)
        cls_token = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_token, x], dim=1)
        attn_mask = self.create_padding_mask(mask, x.device)
        attn_weights = None
        for layer in self.transformer_layers:
            x, attn = layer(x, src_mask=attn_mask, training=training)
            attn_weights = attn
        cls_output = self.norm(x[:, 0, :])
        if attn_weights is None:
            attn_weights = torch.zeros(batch_size, self.nhead, x.size(1), x.size(1), device=x.device)
        return cls_output, attn_weights

# CNN+Transformer+Siamese模型
class SiameseCNNTransformerModel(nn.Module):
    def __init__(self, input_dim=4, d_model=256, nhead=16, num_layers=2, window_size=256, overlap_percent=0.25, 
                 cnn_dropout=0.5, attn_dropout=0.3, ffn_dropout=0.2, cls_dropout=0.1, device='cpu'):
        super().__init__()
        self.cnn_layer = CNNLayer(input_dim, d_model, window_size, overlap_percent, cnn_dropout)
        self.transformer_layer = TransformerLayer(d_model=512, nhead=nhead, num_layers=num_layers, dropout=attn_dropout, cls_dropout=cls_dropout, device=device)
        self.classifier = FullyConnectedLayer(d_model=512, dropout=cls_dropout)
        self.siamese_head = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.LayerNorm(512),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.LayerNorm(256)
        )
        self.initialize_weights()

    def initialize_weights(self):
        for module in self.siamese_head:
            if isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(module.weight, mode='fan_in', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x, original_lengths, training=True, return_features=False):
        cnn_out, mask = self.cnn_layer(x, original_lengths, training=training)
        transformer_out, attn_weights = self.transformer_layer(cnn_out, mask, original_lengths, training=training)
        if return_features:
            features = self.siamese_head(transformer_out)
            return features, attn_weights
        logits = self.classifier(transformer_out)
        return logits, attn_weights

# 主函数（仅训练）
def main(active_fasta, dormant_fasta, 
         num_threads=128, num_epochs=40, accumulation_steps=1, base_lr=1e-5, 
         window_size=128, overlap_percent=0.25, output_dir="./output", batch_size=4, 
         siamese_weight_peak=1.0, siamese_rise_epochs=10, 
         siamese_decay_epochs=15, alpha=0.5, cnn_dropout=0.2, attn_dropout=0.3, 
         ffn_dropout=0.2, cls_dropout=0.5, joint_training_epochs=30):
    
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Ensured output directory: {output_dir}")

    torch.set_num_threads(num_threads)
    logger.info(f"Set CPU threads to {num_threads}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.init()
        gpu_count = torch.cuda.device_count()
        logger.info(f"Initialized CUDA, detected {gpu_count} GPUs")
        if gpu_count > 1:
            logger.info("Using DataParallel for multi-GPU training")
    else:
        logger.info("CUDA unavailable, falling back to CPU")
    logger.info(f"Primary device: {device}")

    # 预处理数据
    train_sequences, train_labels, train_records = preprocess_sequences(
        active_fasta, dormant_fasta, vocab_map, vocab_array, num_threads=num_threads
    )

    # 创建数据集和数据加载器
    train_dataset = ProphageDataset(train_sequences, train_labels)
    siamese_dataset = SiameseDataset(train_sequences, train_labels)

    def worker_init_fn(worker_id):
        np.random.seed(47003 + worker_id)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                                 collate_fn=supervised_collate_fn, num_workers=2, 
                                 worker_init_fn=worker_init_fn)
    siamese_dataloader = DataLoader(siamese_dataset, batch_size=batch_size, shuffle=True, 
                                   collate_fn=siamese_collate_fn, num_workers=2, 
                                   worker_init_fn=worker_init_fn)

    # 初始化模型
    model = SiameseCNNTransformerModel(
        input_dim=input_dim, d_model=256, nhead=16, num_layers=2, window_size=window_size, 
        overlap_percent=overlap_percent, cnn_dropout=cnn_dropout, attn_dropout=attn_dropout, 
        ffn_dropout=ffn_dropout, cls_dropout=cls_dropout, device=device
    ).to(device)

    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model, device_ids=[0, 1])

    # 优化器设置
    base_lrs = [base_lr * 2.0, base_lr, base_lr * 1.5, base_lr * 0.5]
    eta_mins = [base_lr * 0.2, base_lr * 0.1, base_lr * 0.15, base_lr * 0.05]
    optimizer = optim.AdamW([
        {'params': model.module.cnn_layer.parameters() if isinstance(model, nn.DataParallel) else model.cnn_layer.parameters(), 'lr': base_lrs[0]},
        {'params': model.module.transformer_layer.parameters() if isinstance(model, nn.DataParallel) else model.transformer_layer.parameters(), 'lr': base_lrs[1]},
        {'params': model.module.siamese_head.parameters() if isinstance(model, nn.DataParallel) else model.siamese_head.parameters(), 'lr': base_lrs[2]},
        {'params': model.module.classifier.parameters() if isinstance(model, nn.DataParallel) else model.classifier.parameters(), 'lr': base_lrs[3]}
    ], weight_decay=5e-2)

    supervised_criterion = FocalLoss(alpha=alpha, gamma=2.0).to(device)
    siamese_criterion = ContrastiveLoss(margin=0.3).to(device)

    torch.cuda.empty_cache()
    logger.info(f"Cleared GPU memory before training")

    # 学习率调度器初始化
    total_batches = len(train_dataloader)
    total_steps = (num_epochs * total_batches) // accumulation_steps
    warmup_steps = total_steps // 10

    warmup_scheduler = WarmupScheduler(optimizer, warmup_steps, base_lrs, eta_mins, total_steps)

    initial_max_norm = 1.0
    final_max_norm = 5.0
    max_norm_transition_steps = total_steps // 3

    def get_dynamic_max_norm(global_step):
        if global_step <= warmup_steps:
            return initial_max_norm
        elif global_step <= warmup_steps + max_norm_transition_steps:
            progress = (global_step - warmup_steps) / max_norm_transition_steps
            return initial_max_norm + (final_max_norm - initial_max_norm) * progress
        else:
            return final_max_norm

    # 训练循环
    model.train()
    for epoch in range(1, num_epochs + 1):
        epoch_start_time = time.time()
        epoch_supervised_loss = 0
        correct = 0
        total = 0
        active_misclassified = 0
        active_total = 0
        dormant_misclassified = 0
        dormant_total = 0

        # 确定当前阶段
        if epoch <= 5:
            stage = 'pretrain'
            for param in (model.module.classifier.parameters() if isinstance(model, nn.DataParallel) else model.classifier.parameters()):
                param.requires_grad = True
            logger.info(f"Epoch {epoch}: Pretraining stage, all parameters trainable")
        elif epoch <= joint_training_epochs:
            stage = 'joint'
            for param in (model.module.classifier.parameters() if isinstance(model, nn.DataParallel) else model.classifier.parameters()):
                param.requires_grad = False
            for param in (model.module.siamese_head.parameters() if isinstance(model, nn.DataParallel) else model.siamese_head.parameters()):
                param.requires_grad = True
            logger.info(f"Epoch {epoch}: Joint training stage, classifier frozen")
        else:
            stage = 'finetune'
            for param in (model.module.classifier.parameters() if isinstance(model, nn.DataParallel) else model.classifier.parameters()):
                param.requires_grad = True
            for param in (model.module.siamese_head.parameters() if isinstance(model, nn.DataParallel) else model.siamese_head.parameters()):
                param.requires_grad = False
            logger.info(f"Epoch {epoch}: Fine-tuning stage, classifier unfrozen, siamese_head frozen")

        siamese_iterator = iter(siamese_dataloader)
        for batch_idx, (inputs, original_lengths, labels) in enumerate(train_dataloader):
            batch_start_time = time.time()
            inputs, original_lengths, labels = inputs.to(device), original_lengths.to(device), labels.to(device)
            optimizer.zero_grad()

            # 计算监督损失
            outputs, _ = model(inputs, original_lengths, training=True)
            supervised_loss = supervised_criterion(outputs, labels)
            batch_supervised_loss = supervised_loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            active_mask = (labels == 1)
            dormant_mask = (labels == 0)
            active_misclassified += ((predicted != labels) & active_mask).sum().item()
            active_total += active_mask.sum().item()
            dormant_misclassified += ((predicted != labels) & dormant_mask).sum().item()

            # 计算对比损失（仅在联合训练阶段）
            siamese_loss = torch.tensor(0.0, device=device, requires_grad=False)
            batch_siamese_loss_value = 0.0
            if stage == 'joint':
                try:
                    anchor_seq, pair_seq, anchor_lengths, pair_lengths, anchor_labels, pair_labels = next(siamese_iterator)
                except StopIteration:
                    siamese_iterator = iter(siamese_dataloader)
                    anchor_seq, pair_seq, anchor_lengths, pair_lengths, anchor_labels, pair_labels = next(siamese_iterator)
                
                anchor_seq, pair_seq, anchor_lengths, pair_lengths, anchor_labels, pair_labels = (
                    anchor_seq.to(device), pair_seq.to(device), anchor_lengths.to(device),
                    pair_lengths.to(device), anchor_labels.to(device), pair_labels.to(device)
                )

                anchor_feat, _ = model(anchor_seq, anchor_lengths, training=True, return_features=True)
                pair_feat, _ = model(pair_seq, pair_lengths, training=True, return_features=True)
                siamese_loss = siamese_criterion(anchor_feat, pair_feat, pair_labels.float())
                batch_siamese_loss_value = siamese_loss.item()

                del anchor_seq, pair_seq, anchor_lengths, pair_lengths, anchor_feat, pair_feat
                torch.cuda.empty_cache()

            # 动态调整对比损失权重
            def get_dynamic_siamese_weight(epoch):
                if epoch <= 5:
                    return 0.0
                elif epoch <= 5 + siamese_rise_epochs:
                    return 0.5 + (siamese_weight_peak - 0.5) * (epoch - 6) / (siamese_rise_epochs - 1)
                else:
                    return siamese_weight_peak - (siamese_weight_peak - 0.1) * (epoch - (5 + siamese_rise_epochs)) / siamese_decay_epochs

            current_siamese_weight = get_dynamic_siamese_weight(epoch)
            total_loss = supervised_loss + current_siamese_weight * siamese_loss

            # 反向传播
            total_loss.backward()
            current_max_norm = get_dynamic_max_norm((epoch * total_batches + batch_idx + 1) // accumulation_steps)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=current_max_norm)
            optimizer.step()

            # 更新学习率
            warmup_scheduler.step(stage=stage)

            # 日志记录
            batch_time = time.time() - batch_start_time
            if torch.cuda.is_available():
                mem_peak = []
                for i in range(torch.cuda.device_count()):
                    mem_peak.append(torch.cuda.max_memory_allocated(i) / (1024 ** 3))
                gpu_mem_info = f"GPU Mem Peak: GPU0: {mem_peak[0]:.2f} GiB, GPU1: {mem_peak[1]:.2f} GiB" if len(mem_peak) > 1 else f"GPU Mem Peak: GPU0: {mem_peak[0]:.2f} GiB"
            else:
                gpu_mem_info = "GPU Mem Peak: 0.00 GiB"
            seq_lengths = ','.join(map(str, original_lengths.tolist()))
            cnn_lr = optimizer.param_groups[0]['lr']
            transformer_lr = optimizer.param_groups[1]['lr']
            siamese_lr = optimizer.param_groups[2]['lr']
            classifier_lr = optimizer.param_groups[3]['lr']
            logger.info(f"Epoch {epoch}/{num_epochs} | Batch {batch_idx + 1}/{total_batches} | "
                        f"Stage: {stage} | Supervised Loss: {batch_supervised_loss:.4f} | "
                        f"Contrastive Loss: {batch_siamese_loss_value:.4f} | "
                        f"Contrastive Weight: {current_siamese_weight:.4f} | "
                        f"Grad Norm: {grad_norm:.4f} | Max Norm: {current_max_norm:.4f} | "
                        f"Batch Time: {batch_time:.2f}s | LR [CNN: {cnn_lr:.6f}, Transformer: {transformer_lr:.6f}, "
                        f"Siamese: {siamese_lr:.6f}, Classifier: {classifier_lr:.6f}] | "
                        f"Focal Alpha: {alpha:.4f} | {gpu_mem_info} | Seq Lengths: [{seq_lengths}]")

            del outputs, supervised_loss, total_loss, predicted
            torch.cuda.empty_cache()

        epoch_time = time.time() - epoch_start_time
        accuracy = 100. * correct / total
        
        # 保存当前模型
        state = {
            'model_state_dict': model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }
        save_path = os.path.join(output_dir, f"model_epoch_{epoch}.pth")
        torch.save(state, save_path)
        logger.info(f"Model saved to {save_path}, Training Accuracy: {accuracy:.2f}%")
        
        logger.info(f"Epoch {epoch}/{num_epochs} completed | Avg Supervised Loss: {epoch_supervised_loss/total_batches:.4f} | "
                    f"Accuracy: {accuracy:.2f}% | Time: {epoch_time:.2f}s")

    # 保存最终模型
    final_save_path = os.path.join(output_dir, "final_model.pth")
    torch.save(model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict(), final_save_path)
    logger.info(f"Final model saved to {final_save_path}")

    torch.cuda.empty_cache()

if __name__ == "__main__":
    active_fasta = '/ampha/tenant/fafu/private/user/zhb/file/iprophit/train_induce_prophage.fasta'
    dormant_fasta = '/ampha/tenant/fafu/private/user/zhb/file/iprophit/data/train_uninduce_prophage.fasta'
    output_dir = "/ampha/tenant/fafu/private/user/zhb/file/iprophit/TRAIN_MODEL"

    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"CUDA version: {torch.version.cuda}")
        logger.info(f"GPU count: {torch.cuda.device_count()}")

    main(
        active_fasta=active_fasta,
        dormant_fasta=dormant_fasta,
        num_threads=128,
        num_epochs=40,
        accumulation_steps=1,
        base_lr=1e-5,
        window_size=128,
        overlap_percent=0.25,
        output_dir=output_dir,
        batch_size=4,
        siamese_weight_peak=0.8,
        siamese_rise_epochs=10,
        siamese_decay_epochs=15,
        alpha=0.6,
        cnn_dropout=0.2,
        attn_dropout=0.3,
        ffn_dropout=0.2,
        cls_dropout=0.5,
        joint_training_epochs=30
    )
