require 'nn'
local utils=require 'utils'

local MaxPooling = nn.SpatialMaxPooling
local AvgPooling = nn.SpatialAveragePooling
local Convolution = nn.SpatialConvolution
local LeakyReLU   = nn.LeakyReLU
local SBatchNorm = nn.SpatialBatchNormalization
local class_count = net_config.class_count

local iChannels
-- The shortcut layer is either identity or 1x1 convolution
local function shortcut(nInputPlane, nOutputPlane, stride)
    local useConv = shortcutType == 'C' or
     (shortcutType == 'B' and nInputPlane ~= nOutputPlane)
    if useConv then
          -- 1x1 convolution
          return nn.Sequential()
                :add(Convolution(nInputPlane, nOutputPlane, 1, 1, stride, stride))
                :add(SBatchNorm(nOutputPlane))
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
          s:add(SBatchNorm(n))
          s:add(LeakyReLU(0.1, true))
          s:add(Convolution(n,n,3,3, 1,1, 1,1))
          s:add(SBatchNorm(n))

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
 --       b:add(LeakyReLU(0.1, true))
        b:add(nn.SpatialDropout(0.3))
       s:add(b)
      end
  return s
end
--- end of resnet block

--- Resnet blocks
local function ResNetBlock(net, nInputPlane, nOutputPlane, count, stride)
  shortcutType = 'B'
  iChannels = nInputPlane 
  net:add(layer(basicblock, nOutputPlane, count, stride))
end

--- Simple convolution blocks
local function ConvReLU(net, nInputPlane, nOutputPlane, name)
  net:add(nn.SpatialConvolution(nInputPlane, nOutputPlane, 3,3, 1,1, 1,1))
  net:add(nn.ReLU(true))
  return net
end

local function ConvBNLeakyReLU(net, nInputPlane, nOutputPlane, name)
  net:add(nn.SpatialConvolution(nInputPlane, nOutputPlane, 3,3, 1,1, 1,1))
  net:add(SBatchNorm(nOutputPlane))
  net:add(nn.LeakyReLU(0.1, true))
  net:add(nn.SpatialDropout(0.3))
  return net
end

local function ConvBNLeakyReLU7x7(net, nInputPlane, nOutputPlane, name)
  net:add(nn.SpatialConvolution(nInputPlane, nOutputPlane, 7,7, 2,2, 3,3))
  net:add(SBatchNorm(nOutputPlane))
  net:add(nn.LeakyReLU(0.1, true))
  net:add(nn.SpatialDropout(0.3))
  return net
end
---------
--------- Create localization not for spatial transformer input size 224 x 224
---------
function CreateResNet(cinput_planes, class_count)
    local net = nn.Sequential()
    
    ConvBNLeakyReLU7x7(net, cinput_planes,  64, 'local_relu_1_1') -- 112 x 112
    net:add(MaxPooling(3,3, 2,2, 1,1))     
    
    ResNetBlock(net, 64, 64, 2, 1)
    --net:add(MaxPooling(2, 2, 2, 2):ceil())     -- 56  x 56
   
    ResNetBlock(net, 64, 128, 2, 2)
    --net:add(MaxPooling(2, 2, 2, 2):ceil())     -- 28  x 28
    
    ResNetBlock(net, 128, 256, 2, 2)
    --net:add(MaxPooling(2, 2, 2, 2):ceil())     -- 14 x 14 
    
    ResNetBlock(net, 256, 512, 2, 2)
    net:add(AvgPooling(7, 7, 1, 1))            -- 512 features as output    
    
    net:add(nn.View(-1, 512))
    net:add(nn.Linear(512, 512))    
    net:add(nn.BatchNormalization(512))
    net:add(nn.LeakyReLU(0.1, true))
    net:add(nn.Dropout(0.3))
    -- add final 
    local classificator = nn.Linear(512, class_count)
    net:add(classificator)
    
    return net
end

local cnn = CreateResNet(net_config.cinput_planes, class_count)

utils.InitNetwork(cnn)

return cnn