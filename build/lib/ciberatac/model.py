import torch
import torch.nn as nn


def convx(in_channels, out_channels, stride=1, kernel_size=3, dilation=1):
    return nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size,
                     stride=stride, padding=1, bias=False, dilation=dilation)


def normalize(out_channels, normtype="BatchNorm"):
    if normtype == "BatchNorm":
        return nn.BatchNorm1d(out_channels)
    else:
        return nn.GroupNorm(1, out_channels)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1,
                 downsample=None, kernel_size=3,
                 dilation=1, activation="ReLU",
                 normtype="BatchNorm"):
        super(ResidualBlock, self).__init__()
        self.conv1 = convx(in_channels, out_channels,
                           stride=stride, kernel_size=kernel_size,
                           dilation=dilation)
        self.bn1 = normalize(out_channels, normtype)
        if activation == "ReLU":
            self.relu = nn.ReLU()
        elif activation == "LeakyReLU":
            self.relu = nn.LeakyReLU()
        else:
            self.relu = nn.GELU()
        self.conv2 = convx(out_channels, out_channels,
                           kernel_size=kernel_size,
                           dilation=dilation)
        self.bn2 = normalize(out_channels, normtype)
        self.downsample = downsample

    def forward(self, x):
        device = x.device
        dtype = x.dtype
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample:
            residual = self.downsample(x)
        # print("Out: {} Res: {}".format(out.shape, residual.shape))
        if out.shape[2] > residual.shape[2]:
            target = torch.zeros(
                out.shape, dtype=dtype, device=device)
            target[:, :, :residual.shape[2]] = residual
            residual = target
            del target
        elif residual.shape[2] > out.shape[2]:
            target = torch.zeros(
                residual.shape, dtype=dtype, device=device)
            target[:, :, :out.shape[2]] = out
            out = target
            del target
        out += residual
        out = self.relu(out)
        return out


class ResNet1D(nn.Module):
    def __init__(self, block, layers, num_classes=10,
                 dp=0.2, init_conv=16, kernel_size=3,
                 stride=1, filter_rate=2,
                 dilations=[1, 1, 1],
                 pool_type="Average",
                 pool_dim=80, activation="ReLU",
                 inputsize=200000, verbose=False,
                 regression=False,
                 normtype="BatchNorm"):
        super(ResNet1D, self).__init__()
        self.regression = regression
        self.out_dim = 2
        if self.regression:
            self.out_dim = 1
        self.verbose = verbose
        self.layers = layers
        self.activation = activation
        self.pool_dim = pool_dim
        self.dilations = dilations
        self.stride = stride
        self.normtype = normtype
        self.kernel_size = kernel_size
        self.in_channels = init_conv
        # sizeparam = int(self.in_channels * (filter_rate ** len(layers)))
        sizeparam = self.in_channels
        for i in range(len(layers)):
            sizeparam = int(sizeparam * filter_rate)
        self.lin_size = int(
            (sizeparam * pool_dim))
        self.conv = convx(
            4, self.in_channels, kernel_size=self.kernel_size,
            # stride=max(10, int(self.kernel_size / 2)),
            dilation=self.dilations[0])
        self.bn = normalize(self.in_channels, self.normtype)
        # self.pool0 = nn.AdaptiveAvgPool1d(int(inputsize / 2))
        if activation == "ReLU":
            self.relu = nn.ReLU()
        elif activation == "LeakyReLU":
            self.relu = nn.LeakyReLU()
        else:
            self.relu = nn.GELU()
        self.dp1 = nn.Dropout(p=dp)
        self.layer1 = self.make_layer(
            block, self.in_channels,
            layers[0], stride=self.stride,
            dilation=self.dilations[0])
        self.dp2 = nn.Dropout(p=dp)
        self.layer2 = self.make_layer(
            block, int(self.in_channels * filter_rate ** 1),
            layers[1],
            stride=self.stride,
            dilation=self.dilations[1])
        self.dp3 = nn.Dropout(p=dp)
        self.layer3 = self.make_layer(
            block,
            int(self.in_channels * filter_rate ** 2),
            layers[2],
            stride=self.stride,
            dilation=self.dilations[2])
        self.dp4 = nn.Dropout(p=dp)
        if len(layers) == 4:
            self.layer4 = self.make_layer(
                block, int(self.in_channels * filter_rate ** 3),
                layers[3], stride=self.stride,
                dilation=self.dilations[3])
            self.dp5 = nn.Dropout(p=dp)
        if pool_type == "Average":
            self.avg_pool = nn.AdaptiveAvgPool1d(pool_dim)
        else:
            self.avg_pool = nn.AdaptiveMaxPool1d(pool_dim)
        self.fc = nn.Sequential(
            nn.Linear(self.lin_size, int(self.lin_size / 2)),
            nn.ReLU(),
            nn.Dropout(p=dp/4),
            nn.Linear(int(self.lin_size / 2), int(self.lin_size / 4)),
            nn.ReLU(),
            nn.Dropout(p=dp/4),
            nn.Linear(int(self.lin_size / 4), num_classes),
            nn.ReLU())
        self.fc2 = nn.Sequential(
            nn.Linear(num_classes * 3, num_classes * 2),
            nn.ReLU(),
            nn.Dropout(p=dp/4),
            nn.Linear(num_classes * 2, num_classes),
            nn.ReLU(),
            nn.Dropout(p=dp/4),
            nn.Linear(num_classes, self.out_dim),
            nn.ReLU())
        # nn.ReLU())
        # self.dense1 = nn.Linear(num_classes * 3, num_classes * 2)
        # self.dense1_relu = nn.ReLU()
        # self.last_linear = nn.Linear(num_classes * 2, self.out_dim)
        # self.last_relu = nn.ReLU()
        self.dense_scvi = nn.Linear(num_classes, num_classes)
        self.tanh_scvi = nn.Tanh()

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm1d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform(m.weight)
                m.bias.data.fill_(0.01)

    def make_layer(self, block, out_channels, blocks, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                convx(self.in_channels, out_channels, stride=stride,
                      kernel_size=self.kernel_size,
                      dilation=dilation),
                nn.BatchNorm1d(out_channels))
        layers = []
        layers.append(
            block(
                self.in_channels, out_channels, stride, downsample,
                kernel_size=self.kernel_size, dilation=dilation,
                activation=self.activation))
        self.in_channels = out_channels
        for i in range(1, blocks):
            layers.append(
                block(out_channels, out_channels,
                      kernel_size=self.kernel_size,
                      dilation=dilation,
                      activation=self.activation,
                      normtype=self.normtype))
        return nn.Sequential(*layers)

    def forward_rna(self, x):
        device = x.device
        dtype = x.dtype
        out = self.conv(x)
        out = self.bn(out)
        # out = self.pool0(out)
        out = self.relu(out)
        out = self.dp1(out)
        out = self.layer1(out)
        out = self.dp2(out)
        out = self.layer2(out)
        out = self.dp3(out)
        out = self.layer3(out)
        out = self.dp4(out)
        if len(self.layers) == 4:
            out = self.layer4(out)
            out = self.dp5(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        if out.shape[1] != self.lin_size:
            if self.verbose:
                print("Out size is {}".format(out.shape))
                print(
                    "Expected dim 1 size is {}".format(
                        self.lin_size))
            target = torch.zeros(
                (out.shape[0], self.lin_size),
                dtype=dtype, device=device)
            target[:, :out.shape[1]] = out
            out = target
            del target
        out = self.fc(out)
        return out

    def forward_dnase(self, x):
        device = x.device
        dtype = x.dtype
        out = self.conv(x)
        if self.verbose:
            print("Input: {}".format(x.shape))
            print("Output at conv0: {}".format(out.shape))
        out = self.bn(out)
        # out = self.pool0(out)
        out = self.relu(out)
        out = self.dp1(out)
        out = self.layer1(out)
        out = self.dp2(out)
        if self.verbose:
            print("Output at conv1: {}".format(out.shape))
        out = self.layer2(out)
        out = self.dp3(out)
        if self.verbose:
            print("Output at conv2: {}".format(out.shape))
        out = self.layer3(out)
        out = self.dp4(out)
        if self.verbose:
            print("Output at conv3: {}".format(out.shape))
        if len(self.layers) == 4:
            out = self.layer4(out)
            out = self.dp5(out)
            if self.verbose:
                print("Output at conv4: {}".format(out.shape))
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        if out.shape[1] != self.lin_size:
            target = torch.zeros(
                (out.shape[0], self.lin_size),
                dtype=dtype, device=device)
            target[:, :out.shape[1]] = out
            out = target
            del target
        out = self.fc(out)
        return out

    def forward(self, x1, x2, scvi):
        # midpos = int(x1.shape[2] / 2)
        # x1[:, :, (midpos - 500):(midpos + 500)] = 0
        # midval = torch.mean(
        #     x1[:, :, (midpos - 100):(midpos + 100)],
        #     dim=(1, 2))
        # midval = midval.reshape(x1.shape[0], 1)
        out1 = self.forward_dnase(x1)
        # print("out1.shape is {}".format(out1.shape))
        out2 = self.forward_rna(x2)
        scvi_weighted = self.tanh_scvi(
            self.dense_scvi(scvi))
        # out1 += midval
        combined = torch.cat(
            (out1, out2, scvi_weighted), dim=1)
        if self.verbose:
            print("DNase-seq embedding shape: {}".format(out1.shape))
            print("Combined output: {}".format(combined.shape))
        # combined += midval
        out = self.fc2(combined)
        # out = self.dense1(combined)
        # out = self.dense1_relu(out)
        # out = self.last_linear(out)
        # if self.regression:
        #     out = self.last_relu(out)
        return out, out2


def _resnet(block, layers, **kwargs):
    model = ResNet1D(block, layers, **kwargs)
    return model


def resnet18(**kwargs):
    return _resnet(ResidualBlock, [2, 2, 2, 2],
                   **kwargs)


def resnet34(**kwargs):
    return _resnet(ResidualBlock, [3, 4, 6, 3],
                   **kwargs)


def resnet101(**kwargs):
    return _resnet(ResidualBlock, [3, 4, 23, 3],
                   **kwargs)


def resnetbase(**kwargs):
    return _resnet(ResidualBlock, [1, 1, 1, 1],
                   **kwargs)


if __name__ == "__main__":
    net = ResNet1D(
        ResidualBlock, [1, 1, 1])
    print(net)
    input1 = torch.rand(
        8, 4, 20000)
    input2 = torch.rand(
        8, 4, 20000)
    input3 = torch.rand(
        8, 10)
    out = net(input1, input2, input3)
    print("Input shape:", input1.shape)
    print("Out shape:", out[0].shape)
