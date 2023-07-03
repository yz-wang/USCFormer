import warnings
import torch.nn.functional
from timm.models.layers import to_2tuple, trunc_normal_

from models.Base_networks import *
from models.RDB import RDB
from models.SwinTransformer import SwinTransformerLayer
from models.DepthwiseSeparableConv import Depthwise_separable_conv

layer_scale = False
init_value = 1e-6

class ConvStream4x(nn.Module):
    """
    Use ConvStream module to process local information of input image

    Input:
        - x: (B, 3, H, W)
    Output:
        - result: (B, 32, H, W)
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size = 2, stride = 2, padding = 0, bias = False)
        self.gelu = nn.GELU()
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size = 2, stride = 2, padding = 0, bias = False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.init_weight()

    def init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.gelu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        result = self.bn3(self.conv3(x))
        return result

class ConvStream2x(nn.Module):
    """
    Use ConvStream module to process local information of input image

    Input:
        - x: (B, 3, H, W)
    Output:
        - result: (B, 32, H, W)
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size = 2, stride = 2, padding = 0, bias = False)
        self.gelu = nn.GELU()
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size = 3, stride = 1, padding = 1, bias = False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.init_weight()

    def init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.gelu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        result = self.bn3(self.conv3(x))
        return result

class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        self.H, self.W = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        self.num_patches = self.H * self.W
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.norm = nn.LayerNorm(embed_dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        # pdb.set_trace()
        x = self.proj(x)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        return x, H, W

class EncoderTransformer(nn.Module):
    def __init__(self, img_size=224, in_chans=3, num_classes=1000, embed_dims=[96, 192, 384, 768],
                 num_heads=[3, 6, 12, 24], depths=[2, 2, 6, 2], window_size = 7):
        super().__init__()
        self.num_classes = num_classes
        self.depths = depths


        # Conv stem definitions
        self.Conv_stem1 = ConvStream4x(in_chans, embed_dims[0])
        self.Conv_stem2 = ConvStream2x(embed_dims[0], embed_dims[1])
        self.Conv_stem3 = ConvStream2x(embed_dims[1], embed_dims[2])
        self.Conv_stem4 = ConvStream2x(embed_dims[2], embed_dims[3])


        # patch embedding definitions
        self.patch_embed1 = PatchEmbed(img_size=img_size, patch_size=7, stride=4, in_chans=in_chans,
                                              embed_dim=embed_dims[0])
        self.patch_embed2 = PatchEmbed(img_size=img_size // 4, patch_size=3, stride=2, in_chans=embed_dims[0],
                                              embed_dim=embed_dims[1])
        self.patch_embed3 = PatchEmbed(img_size=img_size // 8, patch_size=3, stride=2, in_chans=embed_dims[1],
                                              embed_dim=embed_dims[2])
        self.patch_embed4 = PatchEmbed(img_size=img_size // 16, patch_size=3, stride=2, in_chans=embed_dims[2],
                                              embed_dim=embed_dims[3])

        self.block1 = nn.Sequential(*[
            SwinTransformerLayer(
                dim=embed_dims[0], num_heads=num_heads[0], window_size=window_size, shift_size=0 if (i % 2 == 0) else window_size // 2)
            for i in range(depths[0])])

        self.block2 = nn.Sequential(*[
            SwinTransformerLayer(
                dim=embed_dims[1], num_heads=num_heads[1], window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2)
            for i in range(depths[1])])

        self.block3 = nn.Sequential(*[
            SwinTransformerLayer(
                dim=embed_dims[2], num_heads=num_heads[2], window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2)
            for i in range(depths[2])])

        self.block4 = nn.Sequential(*[
            SwinTransformerLayer(
                dim=embed_dims[3], num_heads=num_heads[3], window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2)
            for i in range(depths[3])])
        self.norm = nn.BatchNorm2d(embed_dims[-1])

        self.Conv1 = nn.Conv2d(embed_dims[0], embed_dims[0], kernel_size=3, stride=1, padding=1, bias=False)
        self.Conv2 = nn.Conv2d(embed_dims[1], embed_dims[1], kernel_size=3, stride=1, padding=1, bias=False)
        self.Conv3 = nn.Conv2d(embed_dims[2], embed_dims[2], kernel_size=3, stride=1, padding=1, bias=False)
        self.Conv4 = nn.Conv2d(embed_dims[3], embed_dims[3], kernel_size=3, stride=1, padding=1, bias=False)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def reset_drop_path(self, drop_path_rate):
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(self.depths))]
        cur = 0
        for i in range(self.depths[0]):
            self.block1[i].drop_path.drop_prob = dpr[cur + i]

        cur += self.depths[0]
        for i in range(self.depths[1]):
            self.block2[i].drop_path.drop_prob = dpr[cur + i]

        cur += self.depths[1]
        for i in range(self.depths[2]):
            self.block3[i].drop_path.drop_prob = dpr[cur + i]

        cur += self.depths[2]
        for i in range(self.depths[3]):
            self.block4[i].drop_path.drop_prob = dpr[cur + i]

    def forward_features(self, x):
        B = x.shape[0]
        outs = []

        # stage 1
        # Conv branch
        conv1 = self.Conv_stem1(x)

        # Transformer branch
        # B,C,W,H
        x1, H1, W1 = self.patch_embed1(x)
        Tran1 = self.block1(x1)

        #stage1_input:(1,3,512,1024), conv1:(1,64,128,256),Tran1:(1,64,128,256)
        y1 = conv1 + Tran1
        y1 = self.Conv1(y1)
        outs.append(y1)

        # stage 2
        # Conv branch
        conv2 = self.Conv_stem2(y1)
        # Transformer branch
        x2, H2, W2 = self.patch_embed2(y1)
        Tran2 = self.block2(x2)

        # stage2_input:(1,64,128,256), conv2:(1,128,64,128),Tran2:(1,128,64,128)
        y2 = conv2 + Tran2
        y2 = self.Conv2(y2)
        outs.append(y2)

        
        # stage 3
        # Conv branch
        conv3 = self.Conv_stem3(y2)
        # Transformer branch
        x3, H3, W3 = self.patch_embed3(y2)
        Tran3 = self.block3(x3)

        # stage3_input:(1,128,64,128), conv3:(1,320,32,64),Tran3:(1,320,32,64)
        y3 = conv3 + Tran3
        y3 = self.Conv3(y3)
        outs.append(y3)


        # stage 4
        # Conv branch
        conv4 = self.Conv_stem4(y3)
        # Transformer branch
        x4, H4, W4 = self.patch_embed4(y3)
        x4 = self.block4(x4)
        Tran4 = self.norm(x4)

        # stage4_input:(1,320,32,64), conv4:(1,512,16,32),Tran4:(1,512,16,32)
        y4 = conv4 + Tran4
        y4 = self.Conv4(y4)
        outs.append(y4)

        return outs

    def forward(self, x):
        x = self.forward_features(x)
        return x

def resize(input,
           size=None,
           scale_factor=None,
           mode='nearest',
           align_corners=None,
           warning=True):
    if warning:
        if size is not None and align_corners:
            input_h, input_w = tuple(int(x) for x in input.shape[2:])
            output_h, output_w = tuple(int(x) for x in size)
            if output_h > input_h or output_w > output_h:
                if ((output_h > 1 and output_w > 1 and input_h > 1
                     and input_w > 1) and (output_h - 1) % (input_h - 1)
                        and (output_w - 1) % (input_w - 1)):
                    warnings.warn(
                        f'When align_corners={align_corners}, '
                        'the output would more aligned if '
                        f'input size {(input_h, input_w)} is `x+1` and '
                        f'out size {(output_h, output_w)} is `nx+1`')
    return F.interpolate(input, size, scale_factor, mode, align_corners)



class Tenc(EncoderTransformer):
    def __init__(self, **kwargs):
        super(Tenc, self).__init__(
            embed_dims=[96, 192, 384, 768], num_heads=[3, 6, 12, 24], depths=[2, 2, 6, 2],window_size=7
            )


class convprojection_base(nn.Module):
    def __init__(self, path=None, **kwargs):
        super(convprojection_base,self).__init__()

        # self.convd32x = UpsampleConvLayer(512, 512, kernel_size=4, stride=2)
        self.convd16x = UpsampleConvLayer(768, 384, kernel_size=4, stride=2)
        self.dense_4 = nn.Sequential(ResidualBlock(384))
        self.convd8x = UpsampleConvLayer(384, 192, kernel_size=4, stride=2)
        self.dense_3 = nn.Sequential(ResidualBlock(192))
        self.convd4x = UpsampleConvLayer(192, 96, kernel_size=4, stride=2)
        self.dense_2 = nn.Sequential(ResidualBlock(96))
        self.convd2x = UpsampleConvLayer(96, 24, kernel_size=4, stride=2)
        self.dense_1 = nn.Sequential(ResidualBlock(24))
        self.convd1x = UpsampleConvLayer(24, 12, kernel_size=4, stride=2)
        self.conv_output = ConvLayer(12, 3, kernel_size=3, stride=1, padding=1)

        self.active = nn.Tanh()        

    def forward(self,x1):
        res16x = self.convd16x(x1[3])

        if x1[2].shape[3] != res16x.shape[3] and x1[2].shape[2] != res16x.shape[2]:
            p2d = (0,-1,0,-1)
            res16x = F.pad(res16x,p2d,"constant",0)
        elif x1[2].shape[3] != res16x.shape[3] and x1[2].shape[2] == res16x.shape[2]:
            p2d = (0,-1,0,0)
            res16x = F.pad(res16x,p2d,"constant",0)
        elif x1[2].shape[3] == res16x.shape[3] and x1[2].shape[2] != res16x.shape[2]:
            p2d = (0,0,0,-1)
            res16x = F.pad(res16x,p2d,"constant",0)

        res8x = self.dense_4(res16x) + x1[2]

        res8x = self.convd8x(res8x)
        res4x = self.dense_3(res8x) + x1[1]
        res4x = self.convd4x(res4x)
        res2x = self.dense_2(res4x) + x1[0]
        res2x = self.convd2x(res2x)
        x = res2x
        x = self.dense_1(x)
        x = self.convd1x(x)

        return x


## The following is the network
class USCFormer(nn.Module):

    def __init__(self, output_nc=3, num_classes=19, seg_dim =32, **kwargs):
        super(USCFormer, self).__init__()

        self.Tenc = Tenc()
        
        self.convproj = convprojection_base()

        #self.clean = ConvLayer(8, 3, kernel_size=3, stride=1, padding=1)
        self.clean = ConvLayer(12, 3, kernel_size=3, stride=1, padding=1)

        #self.active = nn.Tanh()

        # semantic fused
        self.seg_convs = nn.Sequential(
            RDB(num_classes),
            nn.Conv2d(num_classes, seg_dim, 1, bias=True)
        )

        self.coarse_convs = nn.Sequential(
            Depthwise_separable_conv(output_nc, seg_dim)
        )

        self.combine_convs = nn.Sequential(
            nn.Conv2d(seg_dim, seg_dim, 3, 1, 1, bias=True),
            nn.Conv2d(seg_dim, seg_dim, 3, 1, 1, bias=True),
            nn.Conv2d(seg_dim, output_nc, 3, 1, 1, bias=True),
            nn.Tanh()
        )

    def forward(self, x, seg):

        #print(x.shape)
        #print(seg.shape)

        x1 = self.Tenc(x)

        x = self.convproj(x1)

        x = self.clean(x)

        # semantic fused
        f = self.coarse_convs(x) + self.seg_convs(seg)
        x1 = self.combine_convs(f)

        return x1



