import torch
import torch.nn as nn
import torch.nn.functional as F
from mytools import MSELossIgnoreNaNv2,convert_configs


class GatedConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, activation=None):
        super().__init__()
        self.feature_conv = nn.Conv2d(
            in_channels, 2 * out_channels,
            kernel_size, stride, padding, dilation, groups, bias
        )
        if activation is None:
            self.activation = nn.Identity()
        else:
            self.activation = activation

    def forward(self, x, output_gate= False):
        o = self.feature_conv(x)
        f, g = torch.chunk(o, 2, dim=1)
        gate = torch.sigmoid(g)
        out = self.activation(f) * gate     # Element-wise modulation
        if output_gate:
            return out, gate
        return out


class SpatioTemporalLSTMCell(nn.Module):
    def __init__(self, in_channel, num_hidden, img_height, img_width, filter_size, stride, layer_norm):
        super(SpatioTemporalLSTMCell, self).__init__()
        self.num_hidden = num_hidden
        self.padding = filter_size // 2
        self._forget_bias = 1.0

        if layer_norm:
            self.conv_x = nn.Sequential(
                nn.Conv2d(in_channel, num_hidden * 7, kernel_size=filter_size, stride=stride, padding=self.padding, bias=False),
                nn.LayerNorm([num_hidden * 7, img_height, img_width])
            )
            self.conv_h = nn.Sequential(
                nn.Conv2d(num_hidden, num_hidden * 4, kernel_size=filter_size, stride=stride, padding=self.padding, bias=False),
                nn.LayerNorm([num_hidden * 4, img_height, img_width])
            )
            self.conv_m = nn.Sequential(
                nn.Conv2d(num_hidden, num_hidden * 3, kernel_size=filter_size, stride=stride, padding=self.padding, bias=False),
                nn.LayerNorm([num_hidden * 3, img_height, img_width])
            )
            self.conv_o = nn.Sequential(
                nn.Conv2d(num_hidden * 2, num_hidden, kernel_size=filter_size, stride=stride, padding=self.padding, bias=False),
                nn.LayerNorm([num_hidden, img_height, img_width])
            )
        else:
            self.conv_x = nn.Sequential(
                nn.Conv2d(in_channel, num_hidden * 7, kernel_size=filter_size, stride=stride, padding=self.padding, bias=False),
            )
            self.conv_h = nn.Sequential(
                nn.Conv2d(num_hidden, num_hidden * 4, kernel_size=filter_size, stride=stride, padding=self.padding, bias=False),
            )
            self.conv_m = nn.Sequential(
                nn.Conv2d(num_hidden, num_hidden * 3, kernel_size=filter_size, stride=stride, padding=self.padding, bias=False),
            )
            self.conv_o = nn.Sequential(
                nn.Conv2d(num_hidden * 2, num_hidden, kernel_size=filter_size, stride=stride, padding=self.padding, bias=False),
            )
        self.conv_last = nn.Conv2d(num_hidden * 2, num_hidden, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x_t, h_t, c_t, m_t):
        x_concat = self.conv_x(x_t)
        h_concat = self.conv_h(h_t)
        m_concat = self.conv_m(m_t)
        i_x, f_x, g_x, i_x_prime, f_x_prime, g_x_prime, o_x = torch.split(x_concat, self.num_hidden, dim=1)
        i_h, f_h, g_h, o_h = torch.split(h_concat, self.num_hidden, dim=1)
        i_m, f_m, g_m = torch.split(m_concat, self.num_hidden, dim=1)

        i_t = torch.sigmoid(i_x + i_h)
        f_t = torch.sigmoid(f_x + f_h + self._forget_bias)
        g_t = torch.tanh(g_x + g_h)

        delta_c = i_t * g_t
        c_new = f_t * c_t + delta_c

        i_t_prime = torch.sigmoid(i_x_prime + i_m)
        f_t_prime = torch.sigmoid(f_x_prime + f_m + self._forget_bias)
        g_t_prime = torch.tanh(g_x_prime + g_m)

        delta_m = i_t_prime * g_t_prime
        m_new = f_t_prime * m_t + delta_m

        mem = torch.cat((c_new, m_new), 1)
        o_t = torch.sigmoid(o_x + o_h + self.conv_o(mem))
        h_new = o_t * torch.tanh(self.conv_last(mem))

        return h_new, c_new, m_new, delta_c, delta_m


class RNN(nn.Module):
    '''
    Adapted to apply the Mask information
    Input: (Batch_size, total_length, channel * patchsize *patchsize, h, w)
    If apply mask information, input channel = output channel + 1
    Adapted version support unequal input_channel and output_channel.
    The outputs are different when training and testing.
    Linxiao Huang
    2025.11.14
    '''
    def __init__(self, configs, mask_valid):
        """
        :param configs:
        :param mask_valid: (H,W), Ture for valid, False for invalid
        """
        super(RNN, self).__init__()
        configs = convert_configs(configs)
        self.configs = configs
        self.input_frame_channel = configs.patch_size * configs.patch_size * configs.input_channels
        self.output_frame_channel = configs.patch_size * configs.patch_size * configs.output_channels
        self.num_layers = configs.num_layers
        self.num_hidden = [configs.hidden_dim] * configs.num_layers
        cell_list = []

        h = configs.img_height // configs.patch_size
        w = configs.img_width // configs.patch_size
        for i in range(self.num_layers):
            in_channel = self.input_frame_channel if i == 0 else self.num_hidden[i - 1]
            cell_list.append(
                SpatioTemporalLSTMCell(in_channel, self.num_hidden[i], h, w, configs.filter_size, configs.stride,
                                       configs.layer_norm)
            )
        self.cell_list = nn.ModuleList(cell_list)
        self.conv_last = GatedConv2d(self.num_hidden[self.num_layers - 1], self.output_frame_channel, kernel_size=1, stride=1, padding=0,
                                   bias=False ) if configs.gated else (
                        nn.Conv2d(self.num_hidden[self.num_layers - 1], self.output_frame_channel, kernel_size=1, stride=1, padding=0,
                                   bias=False))
        # shared adapter
        adapter_num_hidden = self.num_hidden[0]
        self.adapter = nn.Conv2d(adapter_num_hidden, adapter_num_hidden, 1, stride=1, padding=0, bias=False)

    def forward(self, frames, mask_true, loss_func=nn.MSELoss()):
        # [batch, length, height, width, channel] -> [batch, length, channel, height, width]
        # frames = frames_tensor.permute(0, 1, 4, 2, 3).contiguous()
        # mask_true = mask_true.permute(0, 1, 4, 2, 3).contiguous()

        batch = frames.shape[0]
        height = frames.shape[3]
        width = frames.shape[4]

        next_frames = []
        h_t = []
        c_t = []
        delta_c_list = []
        delta_m_list = []

        decouple_loss = []

        for i in range(self.num_layers):
            zeros = torch.zeros([batch, self.num_hidden[i], height, width]).to(self.configs.device)
            h_t.append(zeros)
            c_t.append(zeros)
            delta_c_list.append(zeros)
            delta_m_list.append(zeros)

        memory = torch.zeros([batch, self.num_hidden[0], height, width]).to(self.configs.device)

        for t in range(self.configs.total_length - 1):
            time_schedule = 1 if self.configs.reverse_schedule else self.configs.input_length
            true_frames = frames[:, t, :self.output_frame_channel]
            if t < time_schedule:
                net = true_frames
            else:
                net = mask_true[t - time_schedule].unsqueeze(0).expand_as(true_frames) * true_frames + \
                          (1 - mask_true[t - time_schedule].unsqueeze(0).expand_as(true_frames)) *  x_gen

            if self.configs.need_mask:
                mask_land = frames[:, t,
                            self.input_frame_channel - self.configs.patch_size ** 2: self.input_frame_channel]
                net = torch.cat((net, mask_land), dim=1)

            h_t[0], c_t[0], memory, delta_c, delta_m = self.cell_list[0](net, h_t[0], c_t[0], memory)
            delta_c_list[0] = F.normalize(self.adapter(delta_c).view(delta_c.shape[0], delta_c.shape[1], -1), dim=2)
            delta_m_list[0] = F.normalize(self.adapter(delta_m).view(delta_m.shape[0], delta_m.shape[1], -1), dim=2)

            for i in range(1, self.num_layers):
                h_t[i], c_t[i], memory, delta_c, delta_m = self.cell_list[i](h_t[i - 1], h_t[i], c_t[i], memory)
                delta_c_list[i] = F.normalize(self.adapter(delta_c).view(delta_c.shape[0], delta_c.shape[1], -1), dim=2)
                delta_m_list[i] = F.normalize(self.adapter(delta_m).view(delta_m.shape[0], delta_m.shape[1], -1), dim=2)

            x_gen = self.conv_last(h_t[self.num_layers - 1])

            next_frames.append(x_gen)
            # decoupling loss
            for i in range(0, self.num_layers):
                decouple_loss.append(
                    torch.mean(torch.abs(torch.cosine_similarity(delta_c_list[i], delta_m_list[i], dim=2))))

        decouple_loss = torch.mean(torch.stack(decouple_loss, dim=0))
        next_frames = torch.stack(next_frames, dim=1)

        if self.training:
            pred = next_frames[:,:, :self.output_frame_channel]
            target = frames[:, 1:, :self.output_frame_channel]
            loss = loss_func(pred, target ) + self.configs.decouple_beta * decouple_loss
        else:
            pred = next_frames[:,self.configs.input_length-1:, :self.output_frame_channel]
            target = frames[:, self.configs.input_length:, :self.output_frame_channel]
            loss = loss_func(pred, target )
        return pred, loss


