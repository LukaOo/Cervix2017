require 'nn'
require 'stn'

local MaxPooling = nn.SpatialMaxPooling
local AvgPooling = nn.SpatialAveragePooling
local Convolution = nn.SpatialConvolution
local LeakyReLU   = nn.LeakyReLU
local SBatchNorm = nn.SpatialBatchNormalization

local iChannels
-- The shortcut layer is either identity or 1x1 convolution
local function shortcut(nInputPlane, nOutputPlane, stride)
    local useConv = shortcutType == 'C' or
     (shortcutType == 'B' and nInputPlane ~= nOutputPlane)
    if useConv then
          -- 1x1 convolution
          return nn.Sequential()
                :add(Convolution(nInputPlane, nOutputPlane, 1, 1, stride, stride))
    elseif nInputPlane ~= nOutputPlane then
          -- Strided, zero-padded identity shortcut
           return nn.Sequential()
                  :add(Avg(1, 1, stride, stride))
                  :add(nn.Concat(2)
                     :add(nn.Identity())
                     :add(nn.MulConstant(0)))
    else
          return nn.Identity()
    end
end

-- The basic residual layer block for 18 and 34 layer network, and the
-- CIFAR networks
local function basicblock(n, stride)
    local nInputPlane = iChannels
    iChannels = n

    local s = nn.Sequential()
          s:add(Convolution(nInputPlane,n,3,3, stride, stride, 1, 1))
          s:add(LeakyReLU(0.1, true))
          s:add(Convolution(n,n,3,3, 1,1, 1,1))

    return nn.Sequential()
     :add(nn.ConcatTable()
        :add(s)
        :add(shortcut(nInputPlane, n, stride)))
     :add(nn.CAddTable(true))
end

-- Creates count residual blocks with specified number of features
local function layer(block, features, count, stride)
  local s = nn.Sequential()
      for i=1,count do
        local b = block(features, i == 1 and stride or 1)
          -- add relu after each residual block except last
        b:add(LeakyReLU(0.1, true))
       s:add(b)
      end
  return s
end
--- end of resnet block

--- Resnet blocks
local function ResNetBlock(net, nInputPlane, nOutputPlane, count)
  shortcutType = 'B'
  iChannels = nInputPlane 
  net:add(layer(basicblock, nOutputPlane, count, 1))
end

--- Simple convolution blocks
local function ConvReLU(net, nInputPlane, nOutputPlane, name)
  net:add(nn.SpatialConvolution(nInputPlane, nOutputPlane, 3,3, 1,1, 1,1))
  net:add(nn.ReLU(true))
  return net
end

---------
--------- Create localization not for spatial transformer input size 224 x 224
---------
function CreateLocalizationResNet(cinput_planes)
    local net = nn.Sequential()
    
    ConvReLU(net, cinput_planes,  32, 'local_relu_1_1')
    ConvReLU(net, 32,  64, 'local_relu_1_2')   -- 224 x 224
    net:add(MaxPooling(2, 2, 2, 2):ceil())     -- 112 x 112
    ResNetBlock(net, 64, 64, 2)
    net:add(MaxPooling(2, 2, 2, 2):ceil())     -- 56  x 56
    ResNetBlock(net, 64, 128, 2)
    net:add(MaxPooling(2, 2, 2, 2):ceil())     -- 28  x 28
    ResNetBlock(net, 128, 128, 2)
    net:add(MaxPooling(2, 2, 2, 2):ceil())     -- 14 x 14 
    ResNetBlock(net, 128, 256, 2)
    net:add(MaxPooling(2, 2, 2, 2):ceil())     -- 7 x 7
    ResNetBlock(net, 256, 512, 2)
    net:add(AvgPooling(7, 7, 1, 1))            -- 512 features as output    
    
    net:add(nn.View(-1, 512))
    net:add(nn.Linear(512, 512))    
    nn.BatchNormalization(512)
    net:add(nn.ReLU(true))
    -- add final 
    local regression = nn.Linear(512, 6)
    regression.weight:zero()
    regression.bias = torch.Tensor({1,0,0,0,1,0})
    net:add(regression)
    
    return net
end

---------
--------- Create localization not for spatial transformer input size 224 x 224
---------
function CreateLocalizationNet(cinput_planes)
    local net = nn.Sequential()
    
    ConvReLU(net, cinput_planes,  16, 'local_relu_1_1') -- 224 x 224
    net:add(MaxPooling(2, 2, 2, 2):ceil())     -- 112 x 112    
    ConvReLU(net, 16,  32, 'local_relu_2_1')  
    net:add(MaxPooling(2, 2, 2, 2):ceil())     -- 56  x 56
    ConvReLU(net, 32,  64, 'local_relu_3_1')  
    net:add(MaxPooling(2, 2, 2, 2):ceil())     -- 28  x 28    
    ConvReLU(net, 64, 128, 'local_relu_4_1')  
    net:add(MaxPooling(2, 2, 2, 2):ceil())     -- 14 x 14 
    ConvReLU(net, 128, 256, 'local_relu_5_1')  
    net:add(MaxPooling(2, 2, 2, 2):ceil())     -- 7 x 7
    ConvReLU(net, 256, 256, 'local_relu_6_1')
    
    net:add(AvgPooling(7, 7, 1, 1))            -- 512 features as output    
    net:add(nn.View(-1, 256))
    net:add(nn.Linear(256, 256))
    nn.BatchNormalization(256)
    net:add(nn.ReLU(true))
    -- add final 
    local regression = nn.Linear(256, 6)
    regression.weight:zero()
    regression.bias = torch.Tensor({1,0,0,0,1,0})
    net:add(regression)
    
    return net
end
-------------
-- Create spatial transformer module
-------------
function CreateSTModule(cinput_planes, image_size, is_resnet)
  
    local localization_network = is_resnet == true and CreateLocalizationResNet(cinput_planes) or CreateLocalizationNet(cinput_planes)

     -- prepare both branches of the st
    local ct = nn.ConcatTable()
    
    -- This branch does not modify the input, just change the data layout to bhwd
    local branch1 = nn.Transpose({3,4},{2,4})

    -- This branch will compute the parameters and generate the grid
    local branch2 = nn.Sequential()
    branch2:add(localization_network)
    -- Here you can restrict the possible transformation with the "use_*" boolean variables
    -- branch2:add( nn.AffineTransformMatrixGenerator(false, true, true) )
    branch2:add(nn.View(-1, 2,3))
    branch2:add(nn.AffineGridGeneratorBHWD(image_size, image_size))

    ct:add(branch1)
    ct:add(branch2)

    ------
    -- Wrap the st in one module
    local st_module = nn.Sequential()
    st_module:add(ct)

    local sampler = nn.BilinearSamplerBHWD()
    st_module:add(sampler)
    
    -- go back to the bdhw layout (used by all default torch modules)
    st_module:add(nn.Transpose({2,4},{3,4}))
    
    return st_module 
end