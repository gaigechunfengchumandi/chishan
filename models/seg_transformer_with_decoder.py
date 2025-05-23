import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1, is_decoder=False):
        super().__init__()
        self.is_decoder = is_decoder
        self.self_attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        if is_decoder:
            self.encoder_attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        if is_decoder:
            self.norm3 = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x, memory=None):
        # Self attention
        attn_output, _ = self.self_attention(x, x, x)
        x = x + attn_output
        x = self.norm1(x)
        
        # Encoder-Decoder attention (for decoder only)
        if self.is_decoder and memory is not None:
            enc_dec_output, _ = self.encoder_attention(x, memory, memory)
            x = x + enc_dec_output
            x = self.norm2(x)
        
        # Feed forward
        ff_output = self.ff(x)
        x = x + ff_output
        x = self.norm3(x) if self.is_decoder else self.norm2(x)
        return x

class TransformerDecoder(nn.Module):
    def __init__(self, num_patches, embed_dim, num_heads, ff_dim, num_layers, dropout=0.1):
        super().__init__()
        self.pos_decoder = nn.Parameter(torch.zeros(num_patches, embed_dim))
        self.layers = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, ff_dim, dropout, is_decoder=True)
            for _ in range(num_layers)
        ])
        self.output_layer = nn.Linear(embed_dim, 128)

    def forward(self, x, memory):
        x = x + self.pos_decoder.unsqueeze(1)
        for layer in self.layers:
            x = layer(x, memory)
        x = self.output_layer(x)
        return x.permute(1, 2, 0)

class TransformerEncoder(nn.Module):
    def __init__(self, num_patches, embed_dim, num_heads, ff_dim, num_layers, dropout=0.1):
        super().__init__()
        self.pos_encoder = nn.Parameter(torch.zeros(num_patches, embed_dim))
        self.layers = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, ff_dim, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, x):
        x = x + self.pos_encoder.unsqueeze(1)
        for layer in self.layers:
            x = layer(x)
        return x

class TransformerSegmentationWithDecoder(nn.Module):
    def __init__(self, input_size=1280, num_classes=5, patch_size=128):
        super().__init__()
        num_patches = input_size // patch_size

        # Encoder部分保持不变
        self.residual_pool = nn.AdaptiveAvgPool1d(num_patches)
        self.residual_conv = nn.Conv1d(1, 128, kernel_size=1)
        self.patch_embed = nn.Sequential(
            nn.Conv1d(1, 512, kernel_size=10, stride=2),
            nn.GELU(),
            nn.Conv1d(512, 512, kernel_size=3, stride=2),
            nn.GELU(),
            nn.Conv1d(512, 512, kernel_size=3, stride=2),
            nn.GELU(),
            nn.Conv1d(512, 512, kernel_size=3, stride=2),
            nn.GELU(),
            nn.Conv1d(512, 512, kernel_size=3, stride=2),
            nn.GELU(),
            nn.Conv1d(512, 512, kernel_size=2, stride=2, padding=1),
            nn.GELU(),
            nn.Conv1d(512, 128, kernel_size=2, stride=2),
            nn.GELU()
        )
        
        # 新增Encoder和Decoder
        self.encoder = TransformerEncoder(
            num_patches=num_patches,
            embed_dim=128,
            num_heads=4,
            ff_dim=256,
            num_layers=3,
            dropout=0.1
        )
        
        self.decoder = TransformerDecoder(
            num_patches=num_patches,
            embed_dim=128,
            num_heads=4,
            ff_dim=256,
            num_layers=3,
            dropout=0.1
        )
        
        self.conv_last = nn.Conv1d(128, out_channels=num_classes, kernel_size=1)
        self.upsampling = nn.Upsample(size=1280, mode='linear')
  
    def forward(self, x):
        residual = self.residual_conv(x)
        residual = self.residual_pool(residual) # （2， 128， 10）
        x = self.patch_embed(x)
        x = x + residual
        x = x.permute(2, 0, 1) # （10， 2， 128）
        
        # 通过encoder和decoder处理
        memory = self.encoder(x) # （10， 2， 128）
        features = self.decoder(x, memory) # （2， 128， 10）
        
        out = self.conv_last(features)
        out = self.upsampling(out)
        return out

if __name__ == "__main__":
    model = TransformerSegmentationWithDecoder()
    writer = SummaryWriter('runs/with_decoder')  # 修改了日志目录以区分不同模型
    test_input = torch.randn(2, 1, 1280)
    output = model(test_input)
    writer.add_graph(model, test_input)
    writer.close()
    print(f"Input shape: {test_input.shape}")
    print(f"Output shape: {output.shape}")
    
    print("\nTensorBoard数据已写入。请运行以下命令查看:")
    print("tensorboard --logdir=/Users/xingyulu/Public/physionet/runs/with_decoder")