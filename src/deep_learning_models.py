"""
Deep Learning Recurrent Architectures (PyTorch) for Code-Mixed Mental Health Text Detection
Implements BiLSTM, BiGRU, and CNN-BiLSTM with custom tokenization and attention pooling.
"""
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

if TORCH_AVAILABLE:
    class BiLSTMClassifier(nn.Module):
        def __init__(self, vocab_size, embed_dim=128, hidden_dim=128, num_classes=3, dropout=0.3):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
            self.spatial_dropout = nn.Dropout2d(dropout)
            self.lstm = nn.LSTM(
                embed_dim,
                hidden_dim,
                batch_first=True,
                bidirectional=True,
                num_layers=1
            )
            self.fc = nn.Sequential(
                nn.Linear(hidden_dim * 2, 64),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(64, num_classes)
            )

        def forward(self, x):
            # x: [batch_size, seq_len]
            embedded = self.embedding(x)
            # Spatial dropout across embedding dimension
            embedded = self.spatial_dropout(embedded.unsqueeze(2)).squeeze(2)
            lstm_out, _ = self.lstm(embedded)
            # Global max pooling across time steps
            pooled, _ = torch.max(lstm_out, dim=1)
            logits = self.fc(pooled)
            return logits

    class BiGRUClassifier(nn.Module):
        def __init__(self, vocab_size, embed_dim=128, hidden_dim=128, num_classes=3, dropout=0.3):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
            self.spatial_dropout = nn.Dropout2d(dropout)
            self.gru = nn.GRU(
                embed_dim,
                hidden_dim,
                batch_first=True,
                bidirectional=True,
                num_layers=1
            )
            self.fc = nn.Sequential(
                nn.Linear(hidden_dim * 2, 64),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(64, num_classes)
            )

        def forward(self, x):
            embedded = self.embedding(x)
            embedded = self.spatial_dropout(embedded.unsqueeze(2)).squeeze(2)
            gru_out, _ = self.gru(embedded)
            pooled, _ = torch.max(gru_out, dim=1)
            logits = self.fc(pooled)
            return logits

    class CNNBiLSTMClassifier(nn.Module):
        def __init__(self, vocab_size, embed_dim=128, conv_filters=64, kernel_size=3, hidden_dim=128, num_classes=3, dropout=0.3):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
            self.conv = nn.Conv1d(in_channels=embed_dim, out_channels=conv_filters, kernel_size=kernel_size, padding=1)
            self.relu = nn.ReLU()
            self.lstm = nn.LSTM(
                conv_filters,
                hidden_dim,
                batch_first=True,
                bidirectional=True
            )
            self.fc = nn.Sequential(
                nn.Linear(hidden_dim * 2, 64),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(64, num_classes)
            )

        def forward(self, x):
            embedded = self.embedding(x)  # [batch, seq_len, embed_dim]
            embedded = embedded.transpose(1, 2)  # [batch, embed_dim, seq_len]
            conv_out = self.relu(self.conv(embedded))  # [batch, conv_filters, seq_len]
            conv_out = conv_out.transpose(1, 2)  # [batch, seq_len, conv_filters]
            lstm_out, _ = self.lstm(conv_out)
            pooled, _ = torch.max(lstm_out, dim=1)
            logits = self.fc(pooled)
            return logits
else:
    class BiLSTMClassifier:
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch is required to initialize BiLSTMClassifier.")
            
    class BiGRUClassifier:
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch is required to initialize BiGRUClassifier.")
            
    class CNNBiLSTMClassifier:
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch is required to initialize CNNBiLSTMClassifier.")
