from torch import nn
import torch
import torch.nn.functional as F

import time

def weight_init_xavier_uniform(submodule):
    if isinstance(submodule, torch.nn.Conv3d):
        torch.nn.init.xavier_normal_(submodule.weight.data)
    elif isinstance(submodule, torch.nn.BatchNorm3d):
        submodule.weight.data.fill_(1.0)
        submodule.bias.data.zero_()

class Conv3DBlock(nn.Module):
    """
    The basic block for double 3x3x3 convolutions in the analysis path
    -- __init__()
    :param in_channels -> number of input channels
    :param out_channels -> desired number of output channels
    :param bottleneck -> specifies the bottlneck block
    -- forward()
    :param input -> input Tensor to be convolved
    :return -> Tensor
    """

    def __init__(self, in_channels, out_channels, bottleneck = False) -> None:
        super(Conv3DBlock, self).__init__()
        self.conv1 = nn.Conv3d(in_channels= in_channels, out_channels=out_channels//2, kernel_size=(3,3,3), padding=1)
        self.bn1 = nn.BatchNorm3d(num_features=out_channels//2)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(in_channels= out_channels//2, out_channels=out_channels, kernel_size=(3,3,3), padding=1)
        self.bn2 = nn.BatchNorm3d(num_features=out_channels)
        self.relu2 = nn.ReLU(inplace=True)
        self.bottleneck = bottleneck
        if not bottleneck:
            self.pooling = nn.MaxPool3d(kernel_size=(2,2,2), stride=2)

        for m in self.modules():
            weight_init_xavier_uniform(m)
    
    
    def forward(self, input):
        res = self.relu1(self.bn1(self.conv1(input)))
        res = self.relu2(self.bn2(self.conv2(res)))
        out = None
        if not self.bottleneck:
            out = self.pooling(res)
        else:
            out = res
        return out, res




class UpConv3DBlock(nn.Module):
    """
    The basic block for upsampling followed by double 3x3x3 convolutions in the synthesis path
    -- __init__()
    :param in_channels -> number of input channels
    :param out_channels -> number of residual connections' channels to be concatenated
    :param last_layer -> specifies the last output layer
    :param num_classes -> specifies the number of output channels for dispirate classes
    -- forward()
    :param input -> input Tensor
    :param residual -> residual connection to be concatenated with input
    :return -> Tensor
    """

    def __init__(self, in_channels, res_channels=0, last_layer=False, num_classes=None) -> None:
        super(UpConv3DBlock, self).__init__()
        assert (last_layer==False and num_classes==None) or (last_layer==True and num_classes!=None), 'Invalid arguments'
        self.upconv1 = nn.ConvTranspose3d(in_channels=in_channels, out_channels=in_channels, kernel_size=(2, 2, 2), stride=2)
        self.relu1 = nn.ReLU(inplace=True)
        self.bn1 = nn.BatchNorm3d(num_features=in_channels//2)
        self.relu2 = nn.ReLU(inplace=True)
        self.bn2 = nn.BatchNorm3d(num_features=in_channels//2)
        self.conv1 = nn.Conv3d(in_channels=in_channels+res_channels, out_channels=in_channels//2, kernel_size=(3,3,3), padding=(1,1,1))
        self.conv2 = nn.Conv3d(in_channels=in_channels//2, out_channels=in_channels//2, kernel_size=(3,3,3), padding=(1,1,1))
        self.last_layer = last_layer
        if last_layer:
            self.conv3 = nn.Conv3d(in_channels=in_channels//2, out_channels=num_classes, kernel_size=(1,1,1))
        
        for m in self.modules():
            weight_init_xavier_uniform(m)
        
    def forward(self, input, residual=None):
        out = self.upconv1(input)
        if residual!=None: 
            out = torch.cat((out, residual), 1)
        out = self.relu1(self.bn1(self.conv1(out)))
        out = self.relu2(self.bn2(self.conv2(out)))
        if self.last_layer: 
            out = self.conv3(out)
            out = F.sigmoid(out)
        return out
        


# class UNet3D(nn.Module):
#     """
#     The 3D UNet model
#     -- __init__()
#     :param in_channels -> number of input channels
#     :param num_classes -> specifies the number of output channels or masks for different classes
#     :param level_channels -> the number of channels at each level (count top-down)
#     :param bottleneck_channel -> the number of bottleneck channels 
#     :param device -> the device on which to run the model
#     -- forward()
#     :param input -> input Tensor
#     :return -> Tensor
#     """
    
#     # def __init__(self, in_channels=1, level_channels=[64, 128, 256, 512], bottleneck_channel=1024) -> None:
#     def __init__(self, in_channels=1, level_channels=[32, 64, 128, 256], bottleneck_channel=512) -> None:
#         super(UNet3D, self).__init__()
#         level_1_chnls, level_2_chnls, level_3_chnls, level_4_chnls = level_channels[0], level_channels[1], level_channels[2], level_channels[3]
#         self.a_block1 = Conv3DBlock(in_channels=in_channels, out_channels=level_1_chnls)
#         self.a_block2 = Conv3DBlock(in_channels=level_1_chnls, out_channels=level_2_chnls)
#         self.a_block3 = Conv3DBlock(in_channels=level_2_chnls, out_channels=level_3_chnls)
#         self.a_block4 = Conv3DBlock(in_channels=level_3_chnls, out_channels=level_4_chnls)

#         self.bottleNeck = Conv3DBlock(in_channels=level_4_chnls, out_channels=bottleneck_channel, bottleneck= True)
#         self.s_block4 = UpConv3DBlock(in_channels=bottleneck_channel, res_channels=level_4_chnls)
#         self.s_block3 = UpConv3DBlock(in_channels=level_4_chnls, res_channels=level_3_chnls)
#         self.s_block2 = UpConv3DBlock(in_channels=level_3_chnls, res_channels=level_2_chnls)
#         self.s_block1 = UpConv3DBlock(in_channels=level_2_chnls, res_channels=level_1_chnls)

#         for m in self.modules():
#             weight_init_xavier_uniform(m)
    
#     def forward(self, input):
#         #Analysis path forward feed
#         # print('UNET INPUT : ', input.shape)
#         # print('UNET INPUT : ', input.shape)

#         # print('input : ', input[0][0][0][0][:3])

#         out, residual_level1 = self.a_block1(input)
#         out, residual_level2 = self.a_block2(out)
#         out, residual_level3 = self.a_block3(out)
#         out, residual_level4 = self.a_block4(out)

#         out, _ = self.bottleNeck(out)

#         #Synthesis path forward feed
#         out = self.s_block4(out, residual_level4)
#         out = self.s_block3(out, residual_level3)
#         out = self.s_block2(out, residual_level2)
#         out = self.s_block1(out, residual_level1)
#         # print('UNET OUTPUT : ', out.shape)
#         # print('UNET OUTPUT : ', out.shape)

#         out =  F.normalize(out, dim=1)

#         return out

class UNet3D(nn.Module):
    # def __init__(self, in_channels=1, level_channels=[64, 128, 256, 512], bottleneck_channel=1024) -> None:
    def __init__(self, in_channels=1, level_channels=[32, 64, 128], bottleneck_channel=256) -> None:
        super(UNet3D, self).__init__()
        level_1_chnls, level_2_chnls, level_3_chnls = level_channels[0], level_channels[1], level_channels[2]
        self.a_block1 = Conv3DBlock(in_channels=in_channels, out_channels=level_1_chnls)
        self.a_block2 = Conv3DBlock(in_channels=level_1_chnls, out_channels=level_2_chnls)
        self.a_block3 = Conv3DBlock(in_channels=level_2_chnls, out_channels=level_3_chnls)
        self.bottleNeck = Conv3DBlock(in_channels=level_3_chnls, out_channels=bottleneck_channel, bottleneck= True)
        self.s_block3 = UpConv3DBlock(in_channels=bottleneck_channel, res_channels=level_3_chnls)
        self.s_block2 = UpConv3DBlock(in_channels=level_3_chnls, res_channels=level_2_chnls)
        self.s_block1 = UpConv3DBlock(in_channels=level_2_chnls, res_channels=level_1_chnls)

        for m in self.modules():
            weight_init_xavier_uniform(m)
    
    def forward(self, input):
        #Analysis path forward feed
        # print('UNET INPUT : ', input.shape)
        # print('UNET INPUT : ', input.shape)

        # print('input : ', input[0][0][0][0][:3])

        out, residual_level1 = self.a_block1(input)
        out, residual_level2 = self.a_block2(out)
        out, residual_level3 = self.a_block3(out)

        out, _ = self.bottleNeck(out)

        #Synthesis path forward feed
        out = self.s_block3(out, residual_level3)
        out = self.s_block2(out, residual_level2)
        out = self.s_block1(out, residual_level1)
        # print('UNET OUTPUT : ', out.shape)
        # print('UNET OUTPUT : ', out.shape)

        out =  F.normalize(out, dim=1)

        return out



        
# if __name__ == '__main__':
#     #Configurations according to the Xenopus kidney dataset
#     model = UNet3D(in_channels=1, num_classes=1).cuda()

#     input_size = (1, 1, 128, 128, 128)
#     rand_input = torch.rand(input_size).cuda()
    
#     print('rand_input : ', rand_input.shape)
#     output = model(rand_input)
#     print('output : ', output.shape)

#     # start_time = time.time()
#     # summary(model=model, input_size=(1, 16, 128, 128), batch_size=-1, device="cpu")
#     # print("--- %s seconds ---" % (time.time() - start_time))