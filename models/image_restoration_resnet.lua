require 'nn'
require 'dpnn'

local MaxPooling = nn.SpatialMaxPooling
local AvgPooling = nn.SpatialAveragePooling
local UpSampling = nn.SpatialUpSamplingBilinear
local Convolution = nn.SpatialConvolution
local LeakyReLU   = nn.LeakyReLU
local SBatchNorm = nn.SpatialBatchNormalization

local baseWidth    = net_config.baseWidth
local cardinality  = net_config.cardinality

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

local function ConvBNReLU(net, nInputPlane, nOutputPlane, stride, name)
  local stride = stride or 1
  net:add(nn.SpatialConvolution(nInputPlane, nOutputPlane, 3,3, stride,stride, 1,1))
  net:add(nn.SpatialBatchNormalization(nOutputPlane))
  net:add(nn.ReLU(true))
  return net
end

local function Conv(net, nInputPlane, nOutputPlane, stride, name)
  local stride = stride or 1
  net:add(nn.SpatialConvolution(nInputPlane, nOutputPlane, 3,3, stride,stride, 1,1))
  return net
end


local function ConvBNLeakyReLU7x7(net, nInputPlane, nOutputPlane, name)
  net:add(nn.SpatialConvolution(nInputPlane, nOutputPlane, 7,7, 2,2, 3,3))
  net:add(SBatchNorm(nOutputPlane))
  net:add(nn.LeakyReLU(0.1, true))
  -- net:add(nn.SpatialDropout(0.3))
  return net
end

function CreateNet(cinput_planes)
    local net = nn.Sequential()
   
    ConvBNLeakyReLU7x7(net, cinput_planes,  64, 'local_relu_1_1') -- 112 x 112
    net:add(MaxPooling(3,3, 2,2, 1,1))     

    ResNetBlock(net, 64, 64, 3, 1)                    -- 56 x 56
    
    ResNetBlock(net, 64, 128, 4, 2)                   -- 28 x 28

    ResNetBlock(net, 128, 256, 6, 2)                  -- 14 x 14
    
    ResNetBlock(net, 256, 512, 3, 2)                  --  7 x 7

    ResNetBlock(net, 512, 256, 3, 1)                  --  
    net:add( UpSampling({oheight=14, owidth=14}) )    --  14 x 14
    
    ResNetBlock(net, 256, 128, 6, 1)                  --  
    net:add( UpSampling({oheight=28, owidth=28}) )    --  28 x 28
    
    ResNetBlock(net, 128, 128, 4, 1)                  --  
    net:add( UpSampling({oheight=56, owidth=56}) )    --  56 x 56
    
    ResNetBlock(net, 128, 64, 3, 1)
    net:add( UpSampling({oheight=112, owidth=112}) )   -- 112 x 112
    
    ResNetBlock(net, 64, 64, 2, 1)
    net:add( UpSampling({oheight=224, owidth=224}) )   -- 224 x  224   
    
    ConvBNReLU(net, 64, 32)    
    ConvBNReLU(net, 32, 3)
       
    return net
end

function CreateNetV2_34(cinput_planes)
    local net = nn.Sequential()
    
    ConvBNReLU(net, cinput_planes, 32, 1)      -- 224x224
    ResNetBlock(net, 32, 64, 2, 2)             -- 112x112
    
    ResNetBlock(net, 64, 128, 3, 2)                     -- 56 x 56
    
    ResNetBlock(net, 128, 256, 4, 2)                    -- 28 x 28

    ResNetBlock(net, 256, 512, 6, 2)                    -- 14 x 14
    
    ResNetBlock(net, 512, 1024, 3, 1)                   --  14 x 14
    
    ResNetBlock(net, 1024, 512, 3, 1)  --  
--    net:add( UpSampling({oheight=14, owidth=14}) )    --  14 x 14
    
    ResNetBlock(net, 512, 256, 6, 1)                  --  
    net:add( UpSampling({oheight=28, owidth=28}) )    --  28 x 28
    
    ResNetBlock(net, 256, 128, 4, 1)                  --  
    net:add( UpSampling({oheight=56, owidth=56}) )    --  56 x 56
    
    ResNetBlock(net, 128, 64, 3, 1)
    net:add( UpSampling({oheight=112, owidth=112}) )   -- 112 x 112
    
    ResNetBlock(net, 64, 64, 2, 1)
    net:add( UpSampling({oheight=224, owidth=224}) )   -- 224 x  224   
    
    ConvBNReLU(net, 64, 32, 1)    
    Conv(net, 32, 3)
    net:add(nn.HardTanh(-0.1, 1.1, true))
       
    return net
end

function CreateNetV3_50(cinput_planes)
    local net = nn.Sequential()
    iChannels = 64
    ConvBNReLU(net, cinput_planes, 64, 1)      -- 224x224   
    ResNetBNBlock(net, 64, 2, 2)             -- 112x112
    
    ResNetBNBlock(net, 128, 3, 2)                     -- 56 x 56
    
    ResNetBNBlock(net, 256, 4, 2)                    -- 28 x 28

    ResNetBNBlock(net, 512, 6, 2)                    -- 14 x 14
    
    ResNetBNBlock(net, 512, 6, 1)                  --  
    net:add( UpSampling({oheight=28, owidth=28}) )    --  28 x 28
    
    ResNetBNBlock(net, 256, 4, 1)                  --  
    net:add( UpSampling({oheight=56, owidth=56}) )    --  56 x 56
    
    ResNetBNBlock(net, 128, 3, 1)
    net:add( UpSampling({oheight=112, owidth=112}) )   -- 112 x 112
    
    ResNetBNBlock(net, 64, 2, 1)
    net:add( UpSampling({oheight=224, owidth=224}) )   -- 256x224 x  224
    
    ConvBNReLU(net, 256, 32, 1)    
    Conv(net, 32, 3)
    net:add(nn.HardTanh(-0.1, 1.1, true))
       
    return net
end

function CreateResNet_50(cinput_planes, block_count)
    local net = nn.Sequential()
    local bc = block_count or 6
    iChannels = 64
    ConvBNReLU(net, cinput_planes, 64, 1)      -- 224x224   
    ResNetBNBlock(net, 64, 3, 2)             -- 112x112
    
    ResNetBNBlock(net, 128, 4, 2)                     -- 56 x 56
    
    ResNetBNBlock(net, 256, bc, 2)                    -- 28 x 28

    ResNetBNBlock(net, 512, 3, 2)                    -- 14 x 14
    
    ResNetBNBlock(net, 512, 3, 1)                  --  
    net:add( UpSampling({oheight=28, owidth=28}) )    --  28 x 28
    
    ResNetBNBlock(net, 256, bc, 1)                  --  
    net:add( UpSampling({oheight=56, owidth=56}) )    --  56 x 56
    
    ResNetBNBlock(net, 128, 4, 1)
    net:add( UpSampling({oheight=112, owidth=112}) )   -- 112 x 112
    
    ResNetBNBlock(net, 64, 3, 1)
    net:add( UpSampling({oheight=224, owidth=224}) )   -- 256x224 x  224
    
    ConvBNReLU(net, 256, 32, 1)    
    Conv(net, 32, 3)
    net:add(nn.HardTanh(-0.1, 1.1, true))
       
    return net
end

function CreateResNetXt_50(cinput_planes)
    local net = nn.Sequential()
    iChannels = 64
    ConvBNReLU(net, cinput_planes, 64, 1)
    
    ResNetBNBlock_B(net, 64, 2, 2)               -- 56x56
    
    ResNetBNBlock_B(net, 128, 3, 2)                -- 28 x 28
    
    ResNetBNBlock_B(net, 256, 4, 2)                    -- 14 x 14 

    ResNetBNBlock_B(net, 512, 4, 2)                    -- 7 x 7
    
    ResNetBNBlock_B(net, 512, 4, 1)                  --  
    net:add( UpSampling({oheight=28, owidth=28}) )    --  14 x 14
    
    ResNetBNBlock_B(net, 256, 4, 1)                  --  
    net:add( UpSampling({oheight=56, owidth=56}) )     --  28 x 28
    
    ResNetBNBlock_B(net, 128, 3, 1)
    net:add( UpSampling({oheight=112, owidth=112}) )    -- 56 x 56
    
    ResNetBNBlock_B(net, 64, 2, 1)
    net:add( UpSampling({oheight=224, owidth=224}) )   --  112 x 112
    
    ConvBNReLU(net, 256, 32, 1)
    Conv(net, 32, 3)
    
    net:add(nn.HardTanh(-0.1, 1.1, true))
          
    return net
end


local cnn = nil

if  net_config.resnet == 'xt_50' then
  cnn = CreateResNetXt_50(3)
else
  if  net_config.resnet == '50' then
      cnn = CreateResNet_50(3, 6)  -- resnet 50 autoencoder
  else
      if  net_config.resnet == '101' then
        cnn = CreateResNet_50(3, 23)  -- resnet 50 autoencoder
      else
        cnn = CreateNetV3_50(3)
      end
  end
end

return cnn