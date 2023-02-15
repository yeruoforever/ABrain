import torch.nn as nn
import torch.nn.functional as F
import torch



class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv3d(2, 1, kernel_size=3, padding=1)
        self.sigmoid = nn.Sigmoid()
    def forward(self, inputs):

        max_out, _ = torch.max(inputs, dim=1)
        max_out = max_out[:, None, :, :, :]
        avg_out = torch.mean(inputs, dim=1)[:, None, :, :, :]

        x = torch.cat([max_out, avg_out], dim=1)

        x = self.conv1(x)
        att = self.sigmoid(x)
        inputs = att * inputs
        return inputs

class ChannelAttention(nn.Module):
    def __init__(self, hidden_size=64, s=4):
        super(ChannelAttention, self).__init__()

        self.max_pool = nn.AdaptiveMaxPool3d((1, 1, 1))
        self.avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))

        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size//s),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size//4, hidden_size)
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs):

        max_x = self.max_pool(inputs)
        avg_x = self.avg_pool(inputs)

        max_x = max_x[None, :, :, 0, 0, 0]
        avg_x = avg_x[None, :, :, 0, 0, 0]

        x = torch.cat([max_x, avg_x], dim=0)

        x = self.mlp(x)

        x = torch.sum(x, dim=0)

        x = self.sigmoid(x)

        attn = x[:, :, None, None, None]

        inputs = attn * inputs

        return inputs


class CBAM(nn.Module):
    def __init__(self, hidden_size=64):
        super(CBAM, self).__init__()

        self.c_model = ChannelAttention(hidden_size)
        self.s_model = SpatialAttention()

    def forward(self, x):

        x = self.c_model(x)
        x = self.s_model(x)

        return x


class conv3d(nn.Module):
    def __init__(self, in_channels, out_channels):
        """
        + Instantiate modules: conv-relu-norm
        + Assign them as member variables
        """
        super(conv3d, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=(5, 5, 1), padding=(2, 2, 0))
        self.relu = nn.PReLU()
        # with learnable parameters
        self.norm = nn.InstanceNorm3d(out_channels, affine=True)

    def forward(self, x):
        return self.relu(self.norm(self.conv(x)))


class conv3d_x3(nn.Module):
    """Three serial convs with a residual connection.
    Structure:
        inputs --> ① --> ② --> ③ --> outputs
                   ↓ --> add--> ↑
    """

    def __init__(self, in_channels, out_channels):
        super(conv3d_x3, self).__init__()
        self.conv_1 = conv3d(in_channels, out_channels)
        self.conv_2 = conv3d(out_channels, out_channels)
        self.conv_3 = conv3d(out_channels, out_channels)
        self.skip_connection = nn.Conv3d(in_channels, out_channels, 1)

    def forward(self, x):
        z_1 = self.conv_1(x)
        z_3 = self.conv_3(self.conv_2(z_1))
        return z_3 + self.skip_connection(x)


class conv3d_x2(nn.Module):
    """Three serial convs with a residual connection.
    Structure:
        inputs --> ① --> ② --> ③ --> outputs
                   ↓ --> add--> ↑
    """

    def __init__(self, in_channels, out_channels):
        super(conv3d_x2, self).__init__()
        self.conv_1 = conv3d(in_channels, out_channels)
        self.conv_2 = conv3d(out_channels, out_channels)
        self.skip_connection = nn.Conv3d(in_channels, out_channels, 1)

    def forward(self, x):
        z_1 = self.conv_1(x)
        z_2 = self.conv_2(z_1)
        return z_2 + self.skip_connection(x)


class conv3d_x1(nn.Module):
    """Three serial convs with a residual connection.
    Structure:
        inputs --> ① --> ② --> ③ --> outputs
                   ↓ --> add--> ↑
    """

    def __init__(self, in_channels, out_channels):
        super(conv3d_x1, self).__init__()
        self.conv_1 = conv3d(in_channels, out_channels)
        self.skip_connection = nn.Conv3d(in_channels, out_channels, 1)

    def forward(self, x):
        z_1 = self.conv_1(x)
        return z_1 + self.skip_connection(x)


class deconv3d_x3(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(deconv3d_x3, self).__init__()
        self.up = deconv3d_as_up(in_channels, out_channels, (2, 2, 1), (2, 2, 1))
        self.lhs_conv = conv3d(out_channels // 2, out_channels)
        self.conv_x3 = nn.Sequential(
            nn.Conv3d(2 * out_channels, out_channels, (5, 5, 1), 1, (2, 2, 0)),
            nn.PReLU(),
            nn.Conv3d(out_channels, out_channels, (5, 5, 1), 1, (2, 2, 0)),
            nn.PReLU(),
            nn.Conv3d(out_channels, out_channels, (5, 5, 1), 1, (2, 2, 0)),
            nn.PReLU(),
        )

    def forward(self, lhs, rhs):
        rhs_up = self.up(rhs)
        lhs_conv = self.lhs_conv(lhs)
        rhs_add = torch.cat((rhs_up, lhs_conv), dim=1)
        return self.conv_x3(rhs_add) + rhs_up


class deconv3d_x2(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(deconv3d_x2, self).__init__()
        self.up = deconv3d_as_up(in_channels, out_channels, (2, 2, 1), (2, 2, 1))
        self.lhs_conv = conv3d(out_channels // 2, out_channels)
        self.conv_x2 = nn.Sequential(
            nn.Conv3d(2 * out_channels, out_channels, (5, 5, 1), 1, (2, 2, 0)),
            nn.PReLU(),
            nn.Conv3d(out_channels, out_channels, (5, 5, 1), 1, (2, 2, 0)),
            nn.PReLU(),
        )

    def forward(self, lhs, rhs):
        rhs_up = self.up(rhs)
        lhs_conv = self.lhs_conv(lhs)
        rhs_add = torch.cat((rhs_up, lhs_conv), dim=1)
        return self.conv_x2(rhs_add) + rhs_up


class deconv3d_x1(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(deconv3d_x1, self).__init__()
        self.up = deconv3d_as_up(in_channels, out_channels, (2, 2, 1), (2, 2, 1))
        self.lhs_conv = conv3d(out_channels // 2, out_channels)
        self.conv_x1 = nn.Sequential(
            nn.Conv3d(2 * out_channels, out_channels, (5, 5, 1), 1, (2, 2, 0)),
            nn.PReLU(),
        )

    def forward(self, lhs, rhs):
        rhs_up = self.up(rhs)
        lhs_conv = self.lhs_conv(lhs)
        rhs_add = torch.cat((rhs_up, lhs_conv), dim=1)
        return self.conv_x1(rhs_add) + rhs_up


def conv3d_as_pool(in_channels, out_channels, kernel_size=(2, 2, 1), stride=(2, 2, 1)):
    return nn.Sequential(
        nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding=0),
        nn.PReLU())


def deconv3d_as_up(in_channels, out_channels, kernel_size=2, stride=2):
    return nn.Sequential(
        nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride),
        nn.PReLU()
    )


class softmax_out(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(softmax_out, self).__init__()
        self.conv_1 = nn.Conv3d(in_channels, out_channels, kernel_size=(5, 5, 1), padding=(2, 2, 0))
        self.conv_2 = nn.Conv3d(out_channels, out_channels, kernel_size=1, padding=0)

    def forward(self, x):
        """Output with shape [batch_size, 1, depth, height, width]."""
        # Do NOT add normalize layer, or its values vanish.
        y_conv = self.conv_2(self.conv_1(x))
        return nn.Sigmoid()(y_conv)


class VNet(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels
        ):
        super(VNet, self).__init__()
        self.conv_1 = conv3d_x1(in_channels, 16)
        self.pool_1 = conv3d_as_pool(16, 32)
        self.conv_2 = conv3d_x2(32, 32)
        self.pool_2 = conv3d_as_pool(32, 64)
        self.conv_3 = conv3d_x3(64, 64)
        self.pool_3 = conv3d_as_pool(64, 128)
        self.conv_4 = conv3d_x3(128, 128)
        self.pool_4 = conv3d_as_pool(128, 256)

        self.bottom = conv3d_x3(256, 256)

        self.attention = CBAM(hidden_size=256)

        self.deconv_4 = deconv3d_x3(256, 256)
        self.deconv_3 = deconv3d_x3(256, 128)
        self.deconv_2 = deconv3d_x2(128, 64)
        self.deconv_1 = deconv3d_x1(64, 32)

        self.ms_conv1 = nn.Sequential(
            nn.Conv3d(1, 32, (2, 2, 1), (2, 2, 1)),
            nn.Conv3d(32, 32, (3, 3, 1), padding=(1, 1, 0))
        )
        self.norm1 = nn.BatchNorm3d(32)

        self.ms_conv2 = nn.Sequential(
            nn.Conv3d(1, 64, (4, 4, 1), (4, 4, 1)),
            nn.Conv3d(64, 64, (3, 3, 1), padding=(1, 1, 0))
        )
        self.norm2 = nn.BatchNorm3d(64)

        self.ms_conv3 = nn.Sequential(
            nn.Conv3d(1, 128, (8, 8, 1), (8, 8, 1)),
            nn.Conv3d(128, 128, (3, 3, 1), padding=(1, 1, 0))
        )
        self.norm3 = nn.BatchNorm3d(128)

        self.ms_conv4 = nn.Sequential(
            nn.Conv3d(1, 256, (16, 16, 1), (16, 16, 1)),
            nn.Conv3d(256, 256, (3, 3, 1), padding=(1, 1, 0))
        )
        self.norm4 = nn.BatchNorm3d(256)

        self.dsconv1 = nn.Sequential(
            nn.ConvTranspose3d(256, 256, (8, 8, 1), (8, 8, 1)),
            nn.Conv3d(256, out_channels, (3, 3, 1), padding=(1, 1, 0)),
            nn.Sigmoid(),
        )

        self.dsconv2 = nn.Sequential(
            nn.ConvTranspose3d(128, 128, (4, 4, 1), (4, 4, 1)),
            nn.Conv3d(128, out_channels, (3, 3, 1), padding=(1, 1, 0)),
            nn.Sigmoid(),
        )

        self.dsconv3 = nn.Sequential(
            nn.ConvTranspose3d(64, 64, (2, 2, 1), (2, 2, 1)),
            nn.Conv3d(64, out_channels, (3, 3, 1), padding=(1, 1, 0)),
            nn.Sigmoid(),
        )

        self.out = softmax_out(32, out_channels)

    def forward(self, x):
        conv_1 = self.conv_1(x)
        pool = self.pool_1(conv_1)

        ms_x1 = self.ms_conv1(x)
        pool = pool + ms_x1
        pool = self.norm1(pool)
        conv_2 = self.conv_2(pool)
        pool = self.pool_2(conv_2)

        ms_x2 = self.ms_conv2(x)
        pool = pool + ms_x2
        pool = self.norm2(pool)
        conv_3 = self.conv_3(pool)
        pool = self.pool_3(conv_3)

        ms_x3 = self.ms_conv3(x)
        pool = pool + ms_x3
        pool = self.norm3(pool)
        conv_4 = self.conv_4(pool)
        pool = self.pool_4(conv_4)

        ms_x4 = self.ms_conv4(x)
        pool = pool + ms_x4
        pool = self.norm4(pool)
        bottom = self.bottom(pool)

        bottom = self.attention(bottom)

        deconv = self.deconv_4(conv_4, bottom)
        ds_out1 = self.dsconv1(deconv)

        deconv = self.deconv_3(conv_3, deconv)
        ds_out2 = self.dsconv2(deconv)

        deconv = self.deconv_2(conv_2, deconv)
        ds_out3 = self.dsconv3(deconv)

        deconv = self.deconv_1(conv_1, deconv)
        out = self.out(deconv)

        return [ds_out1, ds_out2, ds_out3], out

