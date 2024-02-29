from prettytable import PrettyTable
import torch
import torch.nn as nn

class ConvNet(nn.Module):
    def __init__(self, kernel_size = 3, activation_fn = nn.ReLU()):
        super().__init__()

        self.max_pooling_2 = nn.MaxPool3d(kernel_size = 2)

        self.up_sampling_2 = nn.Upsample(scale_factor = 2)

        self.conv64_1_8 = nn.Sequential(
            nn.Conv3d(in_channels = 1, out_channels = 8, kernel_size = kernel_size, padding = 'same'),
            nn.BatchNorm3d(num_features = 8),
            activation_fn
        )

        self.conv64_8_8 = nn.Sequential(
            nn.Conv3d(in_channels = 8, out_channels = 8, kernel_size = kernel_size, padding = 'same'),
            nn.BatchNorm3d(num_features = 8),
            activation_fn
        )

        self.conv32_8_32 = nn.Sequential(
            nn.Conv3d(in_channels = 8, out_channels = 32, kernel_size = kernel_size, padding = 'same'),
            nn.BatchNorm3d(num_features = 32),
            activation_fn
        )

        self.conv32_32_32 = nn.Sequential(
            nn.Conv3d(in_channels = 32, out_channels = 32, kernel_size = kernel_size, padding = 'same'),
            nn.BatchNorm3d(num_features = 32),
            activation_fn
        )

        self.conv16_32_128 = nn.Sequential(
            nn.Conv3d(in_channels = 32, out_channels = 128, kernel_size = kernel_size, padding = 'same'),
            nn.BatchNorm3d(num_features = 128),
            activation_fn
        )

        self.conv16_128_128 = nn.Sequential(
            nn.Conv3d(in_channels = 128, out_channels = 128, kernel_size = kernel_size, padding = 'same'),
            nn.BatchNorm3d(num_features = 128),
            activation_fn
        )

        self.conv8_128_256 = nn.Sequential(
            nn.Conv3d(in_channels = 128, out_channels = 256, kernel_size = kernel_size, padding = 'same'),
            nn.BatchNorm3d(num_features = 256),
            activation_fn
        )

        self.conv8_256_256 = nn.Sequential(
            nn.Conv3d(in_channels = 256, out_channels = 256, kernel_size = kernel_size, padding = 'same'),
            nn.BatchNorm3d(num_features = 256),
            activation_fn
        )

        self.conv16_384_128 = nn.Sequential(
            nn.Conv3d(in_channels = 384, out_channels = 128, kernel_size = kernel_size, padding = 'same'),
            nn.BatchNorm3d(num_features = 128),
            activation_fn
        )

        self.conv32_160_32 = nn.Sequential(
            nn.Conv3d(in_channels = 160, out_channels = 32, kernel_size = kernel_size, padding = 'same'),
            nn.BatchNorm3d(num_features = 32),
            activation_fn
        )

        self.conv64_40_8 = nn.Sequential(
            nn.Conv3d(in_channels = 40, out_channels = 8, kernel_size = kernel_size, padding = 'same'),
            nn.BatchNorm3d(num_features = 8),
            activation_fn
        )

        self.conv64_8_1 = nn.Sequential(
            nn.Conv3d(in_channels = 8, out_channels = 1, kernel_size = kernel_size, padding = 'same'),
            activation_fn
        )

    def forward(self, x):
        x = self.conv64_1_8(x)
        x = self.conv64_8_8(x)
        feature_map_64 = x.detach()
        x = self.max_pooling_2(x)
        x = self.conv32_8_32(x)
        x = self.conv32_32_32(x)
        feature_map_32 = x.detach()
        x = self.max_pooling_2(x)
        x = self.conv16_32_128(x)
        x = self.conv16_128_128(x)
        feature_map_16 = x.detach()
        x = self.max_pooling_2(x)
        x = self.conv8_128_256(x)
        x = self.conv8_256_256(x)
        x = self.up_sampling_2(x)
        x = torch.cat((feature_map_16, x), dim = 1)
        x = self.conv16_384_128(x)
        x = self.conv16_128_128(x)
        x = self.up_sampling_2(x)
        x = torch.cat((feature_map_32, x), dim = 1)
        x = self.conv32_160_32(x)
        x = self.conv32_32_32(x)
        x = self.up_sampling_2(x)
        x = torch.cat((feature_map_64, x), dim = 1)
        x = self.conv64_40_8(x)
        x = self.conv64_8_1(x)
        return x

class ConvNetScalarLabel(nn.Module):
    def __init__(self, kernel_size = 3, activation_fn = nn.ReLU()):
        super().__init__()

        self.layers = nn.ModuleList()
        self.layers.append(self.create_conv_set(1, 2, kernel_size, activation_fn))
        # self.layers.append(self.create_conv_set(2, 2, kernel_size, activation_fn))
        self.layers.append(nn.MaxPool3d(kernel_size = 2))
        self.layers.append(self.create_conv_set(2, 4, kernel_size, activation_fn))
        # self.layers.append(self.create_conv_set(4, 4, kernel_size, activation_fn))
        self.layers.append(nn.MaxPool3d(kernel_size = 2))
        self.layers.append(self.create_conv_set(4, 8, kernel_size, activation_fn))
        # self.layers.append(self.create_conv_set(8, 8, kernel_size, activation_fn))
        self.layers.append(nn.MaxPool3d(kernel_size = 2))
        self.layers.append(self.create_conv_set(8, 32, kernel_size, activation_fn))
        # self.layers.append(self.create_conv_set(32, 32, kernel_size, activation_fn))
        self.layers.append(nn.MaxPool3d(kernel_size = 2))
        self.layers.append(self.create_conv_set(32, 128, kernel_size, activation_fn))
        # self.layers.append(self.create_conv_set(128, 128, kernel_size, activation_fn))
        self.layers.append(nn.MaxPool3d(kernel_size = 2))
        self.layers.append(self.create_conv_set(128, 256, kernel_size, activation_fn))
        # self.layers.append(self.create_conv_set(256, 256, kernel_size, activation_fn))
        self.layers.append(nn.MaxPool3d(kernel_size = 8))

        self.linear_1 = nn.Linear(256, 16)
        self.linear_2 = nn.Linear(16, 1)

    def create_conv_set(self, in_channels, out_channels, kernel_size, activation_fn):
        return nn.Sequential(
            nn.Conv3d(in_channels = in_channels, out_channels = out_channels, kernel_size = kernel_size, padding = 'same'),
            nn.BatchNorm3d(num_features = out_channels),
            activation_fn
        )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = torch.squeeze(x)
        x = self.linear_1(x)
        x = self.linear_2(x)
        return torch.squeeze(x)

# Code below is from https://stackoverflow.com/questions/49201236/check-the-total-number-of-parameters-in-a-pytorch-model
def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params += params
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params