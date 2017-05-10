require 'nn'
require 'dpnn'

local MaxPooling = nn.SpatialMaxPooling
local AvgPooling = nn.SpatialAveragePooling
local UpSampling = nn.SpatialUpSamplingBilinear
local Convolution = nn.SpatialConvolution
local LeakyReLU   = nn.LeakyReLU
local ReLU   = nn.ReLU
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
         :add(LeakyReLU(0.1, true))
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


function CreateResNet_50(cinput_planes, blocks_cnt)
    local net = nn.Sequential()
    local bc = blocks_cnt or 6
    print ('Block count', bc)
    iChannels = 64
    ConvBNReLU(net, cinput_planes, 64, 2)      -- 512x512 ->256
    
    ResNetBNBlock(net, 64, 3, 2)               -- 128x128
    
    ResNetBNBlock(net, 128, 4, 2)                     -- 64 x 64
    
    ResNetBNBlock(net, 256, bc, 2)                    -- 32 x 32

    ResNetBNBlock(net, 512, 3, 2)                    -- 16 x 16
    
    ResNetBNBlock(net, 512, 3, 1)                  --  
    net:add( UpSampling({oheight=32, owidth=32}) )    --  32 x 32
    
    ResNetBNBlock(net, 256, bc, 1)                  --  
    net:add( UpSampling({oheight=64, owidth=64}) )    --  64 x 64
    
    ResNetBNBlock(net, 128, 4, 1)
    net:add( UpSampling({oheight=128, owidth=128}) )    -- 128 x 128
    
    ResNetBNBlock(net, 64, 3, 1)
    net:add( UpSampling({oheight=256, owidth=256}) )   -- 256x256
    ConvBNReLU(net, 256, 64, 1)
    
    net:add( UpSampling({oheight=512, owidth=512}) )   -- 512x512
    ConvBNReLU(net, 64, 32, 1)    
    Conv(net, 32, 1)
    
    if opt.criterion == 'Dice' then
      net:add( nn.HardTanh(0, 1, true) )
    else 
      net:add(nn.Sigmoid())
    end
          
    return net
end

function CreateResNetXt_50(cinput_planes)
    local net = nn.Sequential()
    iChannels = 64
    ConvBNReLU(net, cinput_planes, 64, 2)      -- 512x512 ->256
    
    ResNetBNBlock_B(net, 64, 3, 2)               -- 128x128
    
    ResNetBNBlock_B(net, 128, 4, 2)                     --64 x 64
    
    ResNetBNBlock_B(net, 256, 6, 2)                    -- 32 x 32 

    ResNetBNBlock_B(net, 512, 3, 2)                    -- 16 x 16
    
    ResNetBNBlock_B(net, 512, 3, 1)                  --  
    net:add( UpSampling({oheight=32, owidth=32}) )    --  32 x 32
    
    ResNetBNBlock_B(net, 256, 6, 1)                  --  
    net:add( UpSampling({oheight=64, owidth=64}) )     --  64 x 64
    
    ResNetBNBlock_B(net, 128, 4, 1)
    net:add( UpSampling({oheight=128, owidth=128}) )    -- 128 x 128
    
    ResNetBNBlock_B(net, 64, 3, 1)
    net:add( UpSampling({oheight=256, owidth=256}) )   --  256 x 256
    
    ConvBNReLU(net, 256, 32, 1)
    net:add( UpSampling({oheight=512, owidth=512}) )   -- 512x512
    
   -- ConvBNReLU(net, 64, 32, 1)
    Conv(net, 32, 1)
    
    if opt.criterion == 'Dice' then
      net:add( nn.HardTanh(0, 1, true) )
    else 
      net:add(nn.Sigmoid())
    end
          
    return net
end

local cnn = nil
if  net_config.resnet == 'xt_50' then
  cnn = CreateResNetXt_50(3)
else
  if  net_config.resnet == '101' then
      cnn = CreateResNet_50(3, 12)  -- resnet 101
  else
      cnn = CreateResNet_50(3)      -- resnet 50
  end
end

return cnn