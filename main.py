import os
import time
import torch
import torchaudio
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv
from tqdm import tqdm

from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedShuffleSplit
from torch.nn.utils.rnn import pad_sequence

from transformers import (
    Wav2Vec2FeatureExtractor,  # 仍可用于 WavLM
    WavLMModel,  # 使用 WavLMModel
    WavLMConfig,  # 添加 WavLMConfig 用于加载配置文件
    get_linear_schedule_with_warmup
)

################################################################################
# 1. 数据集定义（采用分层采样，训练阶段动态数据增强不使用缓存）
################################################################################
class UrbanSoundDataset(Dataset):
    def __init__(
            self,
            csv_file='urbansound8k/UrbanSound8K.csv',
            audio_dir='urbansound8k',
            feature_extractor=None,
            split="train",  # "train" 或 "test"
            max_duration_seconds=4.0,  # 此处仅用于记录时长，原始 waveform 保持不变
            augment=False,
            use_cache=True,
            cache_dir='./feature_cache'
    ):
        self.feature_extractor = feature_extractor
        self.audio_dir = audio_dir
        self.max_duration = max_duration_seconds
        self.augment = augment
        self.split = split
        if self.split == "train":
            self.use_cache = False
        else:
            self.use_cache = use_cache
        self.cache_dir = cache_dir

        if self.use_cache and not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir, exist_ok=True)

        print(f"Loading CSV file from {csv_file} ...")
        df = pd.read_csv(csv_file)
        splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        train_idx, test_idx = next(splitter.split(df, df["classID"]))
        selected_indices = train_idx if split == "train" else test_idx

        self.file_paths = []
        self.labels = []
        for idx in selected_indices:
            row = df.iloc[idx]
            fold = row["fold"]
            file_name = row["slice_file_name"]
            full_path = os.path.join(audio_dir, f"fold{fold}", file_name)
            if os.path.isfile(full_path):
                self.file_paths.append(full_path)
                self.labels.append(row["classID"])
            else:
                print(f"File not found: {full_path}")

        unique_labels = sorted(set(self.labels))
        self.label2id = {label: i for i, label in enumerate(unique_labels)}
        self.id2label = {i: label for label, i in self.label2id.items()}
        self.label_indices = [self.label2id[l] for l in self.labels]
        print(f"Loaded {len(self.file_paths)} audio files for {split}.")

    def __len__(self):
        return len(self.file_paths)

    def add_noise(self, waveform, noise_factor=0.005):
        noise = torch.randn(waveform.size()) * noise_factor
        return waveform + noise

    def time_shift(self, waveform, shift_max=0.2):
        shift_amt = int(torch.rand(1).item() * shift_max * waveform.shape[1])
        return torch.roll(waveform, shifts=shift_amt, dims=1)

    def _cache_path(self, idx):
        audio_path = self.file_paths[idx]
        file_name = os.path.basename(audio_path) + ".npz"
        return os.path.join(self.cache_dir, file_name)

    def __getitem__(self, idx):
        if self.use_cache:
            cache_f = self._cache_path(idx)
            if os.path.isfile(cache_f):
                data = np.load(cache_f, allow_pickle=True)
                input_values = torch.tensor(data["input_values"])
                attention_mask = torch.tensor(data["attention_mask"])
                label_id = int(data["label_id"])
                audio_length = float(data["audio_length"])
                return input_values, attention_mask, label_id, audio_length

        audio_path = self.file_paths[idx]
        label_id = self.label_indices[idx]
        waveform, sr = torchaudio.load(audio_path)
        if sr != self.feature_extractor.sampling_rate:
            resampler = torchaudio.transforms.Resample(sr, self.feature_extractor.sampling_rate)
            waveform = resampler(waveform)
            sr = self.feature_extractor.sampling_rate
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        audio_length = waveform.shape[1] / sr

        if self.augment and self.split == "train":
            if torch.rand(1).item() > 0.5:
                waveform = self.add_noise(waveform)
            if torch.rand(1).item() > 0.5:
                waveform = self.time_shift(waveform)

        inputs = self.feature_extractor(
            waveform.squeeze().numpy(),
            sampling_rate=sr,
            return_tensors="pt",
            return_attention_mask=True
        )
        input_values = inputs["input_values"].squeeze(0)
        attention_mask = inputs["attention_mask"].squeeze(0)
        if self.use_cache:
            cache_f = self._cache_path(idx)
            np.savez(cache_f,
                     input_values=input_values.numpy(),
                     attention_mask=attention_mask.numpy(),
                     label_id=label_id,
                     audio_length=audio_length)
        return input_values, attention_mask, label_id, audio_length


################################################################################
# 2. 高级分类头：基于 Transformer 编码器、多头注意力池化与全局平均池化
################################################################################
class WavLMAdvancedClassifier(nn.Module):
    def __init__(self, num_labels, dropout_rate=0.1,
                 freeze_wavlm=True,
                 transformer_layers=4,
                 transformer_heads=8,
                 transformer_dim=512,
                 pool_dim=256):
        super().__init__()
        print("Loading pre-trained WavLM model from local directory...")

        # 使用本地文件路径加载模型
        model_path = os.path.join(os.getcwd(), "pytorch_model.bin")  # 使用当前目录中的路径
        config_path = os.path.join(os.getcwd(), "config.json")  # 使用当前目录中的路径

        # 加载配置文件
        config = WavLMConfig.from_json_file(config_path)

        # 从本地加载模型
        self.wavlm = WavLMModel(config=config)

        self.d_model = self.wavlm.config.hidden_size
        self.wavlm.config.mask_time_prob = 0
        if freeze_wavlm:
            for param in self.wavlm.parameters():
                param.requires_grad = False
            self.frozen = True
            print("WavLM parameters frozen.")

        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=self.d_model,
                nhead=transformer_heads,
                dim_feedforward=transformer_dim
            ),
            num_layers=transformer_layers
        )

        self.attention_pooling = nn.MultiheadAttention(
            embed_dim=self.d_model,
            num_heads=transformer_heads
        )
        self.pooling_query = nn.Parameter(torch.randn(1, 1, self.d_model))
        self.global_avg_pooling = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(self.d_model, pool_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(pool_dim, num_labels)
        )
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, input_values, attention_mask=None, labels=None):
        outputs = self.wavlm(
            input_values=input_values,
            attention_mask=attention_mask
        )
        x = outputs.last_hidden_state
        x = x.permute(1, 0, 2)  # [seq_len, batch, d_model]
        x = self.transformer_encoder(x)
        x = x.permute(1, 0, 2)  # [batch, seq_len, d_model]
        x_transposed = x.permute(1, 0, 2)  # [seq_len, batch, d_model]
        pooling_query = self.pooling_query.repeat(1, x.size(0), 1)
        pooled_output, _ = self.attention_pooling(query=pooling_query,
                                                  key=x_transposed,
                                                  value=x_transposed)
        pooled_output = pooled_output.squeeze(0)  # [batch, d_model]
        pooled_output = self.global_avg_pooling(pooled_output.unsqueeze(-1)).squeeze(-1)
        logits = self.classifier(pooled_output)
        if labels is not None:
            labels = labels.view(-1)
            loss = self.loss_fn(logits, labels)
            return {"loss": loss, "logits": logits}
        return {"logits": logits}



################################################################################
# 3. collate_fn：对批次内变长序列进行填充
################################################################################
def collate_fn(batch):
    input_values = [item[0] for item in batch]
    attention_masks = [item[1] for item in batch]
    labels = [item[2] for item in batch]
    audio_lengths = [item[3] for item in batch]
    input_values_padded = pad_sequence(input_values, batch_first=True, padding_value=0)
    attention_masks_padded = pad_sequence(attention_masks, batch_first=True, padding_value=0)
    labels = torch.tensor(labels, dtype=torch.long)
    return input_values_padded, attention_masks_padded, labels, audio_lengths


################################################################################
# 4. 主训练逻辑：仅在每个 Epoch 结束时打印进度
################################################################################
def main():
    # 分布式训练初始化（若环境变量中存在 RANK 与 WORLD_SIZE，则使用 NCCL 后端）
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        torch.distributed.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        device = torch.device("cuda", local_rank)
        print(f"Distributed training initialized. Local rank: {local_rank}")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Running on GPU." if device.type == "cuda" else "Running on CPU.")

    pin_memory = True if device.type == "cuda" else False

    # 训练参数
    num_epochs = 10
    batch_size = 16
    dropout_rate = 0.1
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("microsoft/wavlm-base-plus")

    # 加载数据集
    train_dataset = UrbanSoundDataset(
        csv_file='urbansound8k/UrbanSound8K.csv',
        audio_dir='urbansound8k',
        feature_extractor=feature_extractor,
        split="train",
        augment=False,
        use_cache=True,
        cache_dir="./feature_cache"
    )
    test_dataset = UrbanSoundDataset(
        csv_file='urbansound8k/UrbanSound8K.csv',
        audio_dir='urbansound8k',
        feature_extractor=feature_extractor,
        split="test",
        augment=False,
        use_cache=True,
        cache_dir="./feature_cache"
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn,
                              pin_memory=pin_memory)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn,
                             pin_memory=pin_memory)

    num_labels = len(train_dataset.label2id)
    print("Initializing WavLM advanced classifier ...")
    model = WavLMAdvancedClassifier(
        num_labels=num_labels,
        dropout_rate=dropout_rate,
        freeze_wavlm=True,
        transformer_layers=4,
        transformer_heads=8,
        transformer_dim=512,
        pool_dim=256
    )
    model.to(device)

    # 多 GPU 支持
    if torch.distributed.is_initialized():
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
    elif torch.cuda.device_count() > 1:
        print(f"Using DataParallel on {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)

    optimizer = AdamW(model.parameters(), lr=1e-5)
    total_steps = len(train_loader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )
    scaler = torch.amp.GradScaler() if device.type == "cuda" else None

    # 打开 CSV 文件写入结果
    with open('training_results.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Epoch', 'Train Loss', 'Train Accuracy', 'Test Accuracy'])  # Header row

        # 记录最佳模型的测试准确率
        best_test_accuracy = 0.0
        best_model_path = "best_model.pth"

        # 开始训练循环，每个 Epoch 结束后统一打印本轮进度
        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0
            correct_preds = 0
            total_preds = 0
            epoch_start_time = time.time()

            for input_values, attention_mask, labels, _ in train_loader:
                input_values = input_values.to(device)
                attention_mask = attention_mask.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()
                if scaler is not None:
                    with torch.amp.autocast(device_type='cuda'):
                        outputs = model(input_values=input_values, attention_mask=attention_mask, labels=labels)
                        loss = outputs["loss"].mean()
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    outputs = model(input_values=input_values, attention_mask=attention_mask, labels=labels)
                    loss = outputs["loss"].mean()
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()

                scheduler.step()
                running_loss += loss.item()
                logits = outputs["logits"]
                _, predicted = torch.max(logits, dim=1)
                correct_preds += (predicted == labels).sum().item()
                total_preds += labels.size(0)

            avg_train_loss = running_loss / len(train_loader)
            train_accuracy = correct_preds / total_preds * 100
            epoch_time = time.time() - epoch_start_time

            # 打印本 Epoch 的训练结果
            print(f"Epoch {epoch + 1}/{num_epochs} finished. "
                  f"Train Loss: {avg_train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%, "
                  f"Time: {epoch_time:.2f}s")

            # 测试阶段：统一在 Epoch 结束后计算测试准确率
            model.eval()
            test_correct_preds = 0
            test_total_preds = 0
            with torch.no_grad():
                for input_values, attention_mask, labels, _ in test_loader:
                    input_values = input_values.to(device)
                    attention_mask = attention_mask.to(device)
                    labels = labels.to(device)
                    outputs = model(input_values=input_values, attention_mask=attention_mask)
                    logits = outputs["logits"]
                    _, predicted = torch.max(logits, dim=1)
                    test_correct_preds += (predicted == labels).sum().item()
                    test_total_preds += labels.size(0)

            test_accuracy = test_correct_preds / test_total_preds * 100
            print(f"Epoch {epoch + 1}: Test Accuracy: {test_accuracy:.2f}%\n")

            # 将结果保存到 CSV 文件中
            writer.writerow([epoch + 1, avg_train_loss, train_accuracy, test_accuracy])

            # 保存最佳模型
            if test_accuracy > best_test_accuracy:
                best_test_accuracy = test_accuracy
                torch.save(model.state_dict(), best_model_path)
                print(f"Saved new best model with Test Accuracy: {best_test_accuracy:.2f}%")

if __name__ == "__main__":
    main()
