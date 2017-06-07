require 'nn'
local utils=require 'utils'
require 'models/gradient_decrease'
require 'models/spatial_bilinear'

local MaxPooling = nn.SpatialMaxPooling
local AvgPooling = nn.SpatialAveragePooling
local Convolution = nn.SpatialConvolution
local LeakyReLU   = nn.LeakyReLU
local SBatchNorm = nn.SpatialBatchNormalization
local class_count = net_config.class_count
local gradient_decrease = net_config.gradiend_decrease or 0.01

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

   -- The aggregated residual transformation bottleneck layer, Form (B)
local function split(nInputPlane, d, c, stride)
      local cat = nn.ConcatTable()
      for i=1,c do
         local s = nn.Sequential()
         s:add(Convolution(nInputPlane,d,1,1,1,1,0,0))
         s:add(SBatchNorm(d))
         s:add(LeakyReLU(0.1, true))
         s:add(Convolution(d,d,3,3,stride,stride,1,1))
         s:add(SBatchNorm(d))
         s:add(LeakyReLU(0.1, true))
         cat:add(s)
      end
      return cat
end
   
local function resnext_bottleneck_B(n, stride)
      local nInputPlane = iChannels
      iChannels = n * 4

      local D = math.floor(n * (baseWidth/64))
      local C = cardinality

      local s = nn.Sequential()
      s:add(split(nInputPlane, D, C, stride))
      s:add(nn.JoinTable(2))
      s:add(Convolution(D*C,n*4,1,1,1,1,0,0))
      s:add(SBatchNorm(n*4))

      return nn.Sequential()
         :add(nn.ConcatTable()
            :add(s)
            :add(shortcut(nInputPlane, n * 4, stride)))
         :add(nn.CAddTable(true))
 --        :add(LeakyReLU(0.1, true))
end

-- The bottleneck residual layer for 50, 101, and 152 layer networks
local function bottleneck(n, stride)
      local nInputPlane = iChannels
      iChannels = n * 4

      local s = nn.Sequential()
      s:add(Convolution(nInputPlane,n,1,1,1,1,0,0))
      s:add(SBatchNorm(n))
      s:add(LeakyReLU(0.1, true))
      s:add(Convolution(n,n,3,3,stride,stride,1,1))
      s:add(SBatchNorm(n))
      s:add(LeakyReLU(0.1, true))
      s:add(Convolution(n,n*4,1,1,1,1,0,0))
      s:add(SBatchNorm(n * 4))

      return nn.Sequential()
         :add(nn.ConcatTable()
            :add(s)
            :add(shortcut(nInputPlane, n * 4, stride)))
         :add(nn.CAddTable(true))
--         :add(LeakyReLU(0.1, true))
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
        b:add(LeakyReLU(0.1, true))
--        b:add(nn.SpatialDropout(0.3))
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

--- Resnet blocks
local function ResNetBNBlock(net, nOutputPlane, count, stride)
  shortcutType = 'B'
  net:add(layer(bottleneck, nOutputPlane, count, stride))
end

--- Resnet blocks
local function ResNetBNBlock_B(net, nOutputPlane, count, stride)
  shortcutType = 'B'
  net:add(layer(resnext_bottleneck_B, nOutputPlane, count, stride))
end

--- Simple convolution blocks
local function ConvReLU(net, nInputPlane, nOutputPlane, name)
  net:add(nn.SpatialConvolution(nInputPlane, nOutputPlane, 3,3, 1,1, 1,1))
  net:add(SBatchNorm(nOutputPlane))
  net:add(nn.LeakyReLU(0.3, true))
  net:add(nn.SpatialDropout(0.3))
  return net
end

local function ConvBNLeakyReLU(net, nInputPlane, nOutputPlane, name)
  net:add(nn.SpatialConvolution(nInputPlane, nOutputPlane, 3,3, 1,1, 1,1))
  net:add(SBatchNorm(nOutputPlane))
  net:add(nn.LeakyReLU(0.1, true))
  --net:add(nn.SpatialDropout(0.3))
  return net
end

local function ConvBNLeakyReLU7x7(net, nInputPlane, nOutputPlane, name)
  net:add(nn.SpatialConvolution(nInputPlane, nOutputPlane, 7,7, 2,2, 3,3))
  net:add(SBatchNorm(nOutputPlane))
  net:add(nn.LeakyReLU(0.1, true))
 -- net:add(nn.SpatialDropout(0.3))
  return net
end

local function ConvBNLeakyReLU7x7_check(net, nInputPlane, nOutputPlane, name)
  net:add(nn.SpatialConvolution(nInputPlane, nOutputPlane, 7,7, 32,32, 3,3))
  net:add(SBatchNorm(nOutputPlane))
  net:add(nn.LeakyReLU(0.1, true))
 -- net:add(nn.SpatialDropout(0.3))
  return net
end

function CreateCheckNet(cinput_planes)
  local net = nn.Sequential()
  ConvBNLeakyReLU7x7_check(net, cinput_planes,  64, 'local_relu_1_1')
  net:add(AvgPooling(7, 7, 1, 1))
  net:add(nn.View(-1, 64))
  net:add(nn.Linear(64, 64)) 
  net:add(nn.BatchNormalization(64))
  return net
end

function CreateCNNNet(cinput_planes)
    local net = nn.Sequential()
    
    ConvReLU(net, cinput_planes,  32, 'local_relu_1_1') -- 224 x 224
    ConvReLU(net, 32,  64, 'local_relu_2_1')
    net:add(MaxPooling(2, 2, 2, 2):ceil())     -- 112 x 112    
    ConvReLU(net, 64,  64, 'local_relu_2_1')
    net:add(MaxPooling(2, 2, 2, 2):ceil())     -- 56  x 56
    ConvReLU(net, 64,  128, 'local_relu_3_1')
    net:add(MaxPooling(2, 2, 2, 2):ceil())     -- 28  x 28    
    ConvReLU(net, 128, 256, 'local_relu_4_1')
    net:add(MaxPooling(2, 2, 2, 2):ceil())     -- 14 x 14 
    ConvReLU(net, 256, 256, 'local_relu_5_1')
    net:add(MaxPooling(2, 2, 2, 2):ceil())     -- 7 x 7
    ConvReLU(net, 256, 256, 'local_relu_6_1')
    net:add(MaxPooling(2, 2, 2, 2):ceil())     -- 4 x 4
    ConvReLU(net, 256, 256, 'local_relu_7_1')
    net:add(nn.View(-1, 256*4*4))
    net:add(nn.Linear(4096, 4096, false))
    
    return net
end

embeding_size = 0
function LoadPretrainedNet(model_file, c_features)
    local MODEL_FILE = './pretrained/'.. model_file
    local class_count = net_config.class_count
    local gradient_decrease = net_config.gradiend_decrease or 0.1
    local  net = torch.load(MODEL_FILE)
    local cf = c_features or 2048
    net:remove() -- remove last Linear
    net:remove() -- remove View
    net:remove() -- remove --AvgPooling
    local fc_dropout = net_config.fc_dropout or 0.7
    
    if fc_dropout > 0 then net:add( nn.SpatialDropout(fc_dropout) ) end
    
    net:add(nn.View(-1, cf, 7 * 7)) -- flat all features map
    if gradient_decrease > 0 then net:add(nn.GradientDecrease(gradient_decrease)) end
    net:add(nn.Transpose({2,3}))
    embeding_size = cf
    -- output 2048 or 512 x 7x7
    return net
end
---------
--------- Create localization not for spatial transformer input size 224 x 224
---------
function CreateResNet_18(cinput_planes, class_count)
    local net = nn.Sequential()
    
    ConvBNLeakyReLU7x7(net, cinput_planes,  64, 'local_relu_1_1') -- 112 x 112
    net:add(MaxPooling(3,3, 2,2, 1,1))     
    
    ResNetBlock(net, 64, 64, 2, 1)
   
    ResNetBlock(net, 64, 128, 2, 2)
    
    ResNetBlock(net, 128, 256, 2, 2)
    
    ResNetBlock(net, 256, 512, 2, 2)

    
    return net
end

function CreateResNet_34(cinput_planes, class_count)
    local net = nn.Sequential()
    
    ConvBNLeakyReLU7x7(net, cinput_planes,  64, 'local_relu_1_1') -- 112 x 112
    net:add(MaxPooling(3,3, 2,2, 1,1))     
    
    ResNetBlock(net, 64, 64, 3, 1) --  56 x 56
   
    ResNetBlock(net, 64, 128, 4, 2) -- 28 x 28
    
    ResNetBlock(net, 128, 256, 6, 2) -- 14 x 14
    
    ResNetBlock(net, 256, 512, 3, 2) -- 7 x 7
 
    
    return net
end

function CreateResNet_XXX(cinput_planes, bc)
    local net = nn.Sequential()
    iChannels = 64
    
    ConvBNLeakyReLU7x7(net, cinput_planes,  64, 'local_relu_1_1') -- 112 x 112
    net:add(MaxPooling(3,3, 2,2, 1,1))     
    
    ResNetBNBlock(net, 64, 3, 1)             -- 56x56
    
    ResNetBNBlock(net, 128, 4, 2)                     -- 28 x 28
    
    ResNetBNBlock(net, 256, bc, 2)                    -- 14 x 14

    ResNetBNBlock(net, 512, 3, 2)                    -- 2048x 7 x 7
    
       
    return net
end

function CreateResNet_50(cinput_planes)
    return CreateResNet_XXX(cinput_planes, 6)
end

function CreateResNet_101(cinput_planes)
    return CreateResNet_XXX(cinput_planes, 23)
end

function CreateResNet_152(cinput_planes)
    return CreateResNet_XXX(cinput_planes, 36)
end

function MakeEmbedingNet(cinput_planes, model_file, c_features)
     if net_config.model_file ~= nil and model_file ~= '' then
       return LoadPretrainedNet(model_file, c_features)
     end
     
     print ("Create net: resnet_" .. net_config.resnet)
     if  net_config.resnet == '50' then
         cnn = CreateResNet_50(cinput_planes)
     else
       if  net_config.resnet == '101' then
           cnn = CreateResNet_101(cinput_planes)  -- resnet 101
       else
         if  net_config.resnet == '152' then
             cnn = CreateResNet_152(cinput_planes)
         else
            cnn = CreateResNet_34(cinput_planes)      -- resnet 34
         end
       end      
   end
   return cnn
end

function CreateResNet(cinput_planes)
   local fc_dropout = net_config.fc_dropout or 0.7
   local net = nn.Sequential()
   local pt = nn.ParallelTable()
   local cnn = nn.Sequential():add( MakeEmbedingNet(cinput_planes, net_config.model_file, net_config.c_features) )
   local emb_l = embeding_size
   local cnn_l = cnn
   local cnn_r =  None
   if net_config.model_file1 == nil then
     cnn_r = cnn:clone('weight','bias', 'gradWeight','gradBias','running_mean','running_std', 'running_var')
   else
     cnn_r = nn.Sequential():add( MakeEmbedingNet(cinput_planes, net_config.model_file1, net_config.c_features1) )
   end
   local emb_r = embeding_size
   pt:add(cnn_l)
   pt:add(cnn_r)
   net:add(pt)
   net:add(nn.SpatialBilinear())
   net:add(nn.View(-1, emb_l * emb_r))
   net:add(nn.SignedSquareRoot())
   net:add(nn.Normalize(2)) -- bilinear module output
   -- classifier
   net:add(nn.Sequential()
         :add(nn.Linear(emb_l * emb_r, net_config.class_count )))

   return net
end

local cnn = CreateResNet(net_config.cinput_planes)

--utils.InitNetwork(cnn)

return cnn