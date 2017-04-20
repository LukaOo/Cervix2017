require 'nn'
require 'cunn'
-- require 'models/gradient_debug'

require 'dpnn'

cudnn = require 'cudnn'

local cinput_planes = net_config.cinput_planes or 3
local image_size    = net_config.image_size    or 512


local Convolution = nn.SpatialConvolution
local Avg = nn.AveragePooling
local ReLU = nn.ReLU
local LeakyReLU = nn.LeakyReLU
local Max = nn.SpatialMaxPooling
local SBatchNorm = nn.SpatialBatchNormalization

-- Will use "ceil" MaxPooling because we want to save as much
-- space as we can
local MaxPooling = nn.SpatialMaxPooling

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
s:add(SBatchNorm(n,1e-3))
--s:add(nn.SpatialDropout(0.3))    
s:add(LeakyReLU(0.1, true))
s:add(Convolution(n,n,3,3, 1,1, 1,1))
s:add(SBatchNorm(n,1e-3))
--s:add(nn.SpatialDropout(0.3))    

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

local function Conv1x1BN(net, nInputPlane, nOutputPlane, name)
  net:add(nn.SpatialConvolution(nInputPlane, nOutputPlane, 1,1))
  net:add(nn.SpatialBatchNormalization(nOutputPlane,1e-3))
  return net
end

local function Conv5x5BN(net, nInputPlane, nOutputPlane, name)
  net:add(nn.SpatialConvolution(nInputPlane, nOutputPlane, 5,5, 1,1, 2,2))
  net:add(nn.SpatialBatchNormalization(nOutputPlane,1e-3))
  return net
end

local function ConvBN(net, nInputPlane, nOutputPlane, name)
  net:add(nn.SpatialConvolution(nInputPlane, nOutputPlane, 3,3, 1,1, 1,1))
  net:add(nn.SpatialBatchNormalization(nOutputPlane,1e-3))
  return net
end

local function DilatedConvBN(net, nInputPlane, nOutputPlane, name)
  net:add(nn.SpatialDilatedConvolution(nInputPlane, nOutputPlane, 3,3, 1,1, 2,2, 2,2))
  net:add(nn.SpatialBatchNormalization(nOutputPlane,1e-3))
  return net
end

local function Conv(nInputPlane, nOutputPlane, name)
  vgg:add(nn.SpatialConvolution(nInputPlane, nOutputPlane, 3,3, 1,1, 1,1))
end

-- building block
local function ConvBNReLU(net, nInputPlane, nOutputPlane, name, is_relu)
  net:add(nn.SpatialConvolution(nInputPlane, nOutputPlane, 3,3, 1,1, 1,1))
  net:add(nn.SpatialBatchNormalization(nOutputPlane,1e-3))
  local relu = is_relu == true and nn.ReLU() or nn.LeakyReLU(0.1, true)
  relu.name = name
  net:add(relu)
  net:add(nn.SpatialDropout(0.3))
  return net
end

local function UpSample(net)
    net:add(nn.SpatialUpSamplingBilinear(2))
    return net
end
--- Resnet blocks
local function ResNetBlock(net, nInputPlane, nOutputPlane, count)
  shortcutType = 'B'
  iChannels = nInputPlane 
  net:add(layer(basicblock, nOutputPlane, count, 1))
  return net
end

local function UBlock(module, fea, dilated)
   if fea ~= nil then
       ConvBN(module, fea, fea)
   end
   local d = dilated or true 
   
   local c1 = nn.Identity() 
   local c2 = Conv1x1BN(nn.Sequential(), fea, fea)
   local c3 = Conv5x5BN(nn.Sequential(), fea, fea)
   local s = nn.Sequential()
      s:add(nn.ConcatTable():add(module):add(c1):add(c2):add(c3))
   s:add(nn.CAddTable(true))
   if fea ~= nil then
      s:add(nn.LeakyReLU(0.1, true))
   end
   s:add(nn.SpatialDropout(0.3))
   return s
end

local unet = nn.Sequential()



--vgg:add( nn.GradientDebug('start_of_resnet') )

ConvBNReLU(unet, cinput_planes, 32, 'relu1_1')  -- for example input size is 1x512x512 -> 32x512x512
ConvBNReLU(unet, 32, 64, 'relu1_2')             --                                      ->64x512x512
unet:add(MaxPooling(2,2, 2, 2):ceil())          -- downsample  input /2 output is 64x256x256

ResNetBlock(unet, 64, 64, 2)

 local ub4 = nn.Sequential()
 ub4:add(MaxPooling(2,2,2,2):ceil())      -- downsample  input /2 output is 64x128x128

 ResNetBlock(ub4, 64, 128, 2)
   local ub3 = nn.Sequential()
   ub3:add(MaxPooling(2,2,2,2):ceil())     -- downsample  input /2 output is 128x64x64  - 48


   ResNetBlock(ub3, 128, 128, 2)
     local ub2 = nn.Sequential()
     ub2:add(MaxPooling(2,2,2,2):ceil())      --  downsample  input /2 output is 128x32x32  - 24


      ResNetBlock(ub2, 128, 256, 2)
    
      local ub1 = nn.Sequential()

        ub1:add(MaxPooling(2,2,2,2):ceil())      -- 256x16x16

        ResNetBlock(ub1, 256, 512, 2)             --512x16x16

        ub1:add( MaxPooling(2,2,2,2):ceil() )
      
        ResNetBlock(ub1, 512, 1024, 2)             --1024x8x8
        
        ResNetBlock(ub1, 1024, 512, 2)             --512x8x8
        
        UpSample(ub1)
      
       ResNetBlock(ub1, 512, 256, 2)             --256x16x16
       UpSample(ub1)
      
      ub2:add(UBlock(ub1, 256))
  

     ResNetBlock(ub2, 256, 128, 2)               -- 512x64x64
    UpSample(ub2)
    
    ub3:add(UBlock(ub2, 128))

    
   ResNetBlock(ub3, 128, 128, 2)                 -- 256x64x64
   
   UpSample(ub3)                            -- 256x128x128
  ub4:add(UBlock(ub3, 128))

 ResNetBlock(ub4, 128, 64, 2)            --128x128x128
 UpSample(ub4)                           -- 128x256x256
 unet:add(UBlock(ub4, 64))

   
ResNetBlock(unet, 64, 64, 2)              -- 64x256x256
UpSample(unet)                            -- 64x512x512

ConvBNReLU(unet, 64, 32, 'relu_5_1')
ConvBN(unet, 32, 1, 'relu_5_1')

-- unet:add( nn.HardTanh(0, 1, true) )
unet:add(nn.Sigmoid())

--vgg:add( nn.GradientDebug('end_of_resnet') )

return unet
