require 'torch'
require 'nn'

local SpatialBilinear, parent = torch.class('nn.SpatialBilinear', 'nn.Module')


function SpatialBilinear:__init(normalize)
  parent.__init(self)
  self.buffer = torch.Tensor()
  self.gradInput = {}
end


function SpatialBilinear:updateOutput(input)
  local C, H, W
  local x1_flat = input[1]
  local x2_flat  = input[2]
  
  N, C = x1_flat:size(1), x1_flat:size(3)
  self.output:resize(N, C, C)
  self.output:bmm( x1_flat:transpose(2, 3), x2_flat)

  return self.output
end


function SpatialBilinear:updateGradInput(input, gradOutput)
 
  self.gradInput[1] = (self.gradInput[1] or input[1].new()):resize(input[1]:size()) 
  self.gradInput[2] = (self.gradInput[2] or input[2].new()):resize(input[2]:size())
  
  local C, H, W
  local x1_flat = input[1]
  local x2_flat = input[2]

  self.buffer:resizeAs(x2_flat)
  self.buffer:bmm( x2_flat, gradOutput:transpose(2, 3))
  self.gradInput[1] = self.buffer:clone()
  
  self.buffer:resizeAs(x1_flat)
  self.buffer:bmm( x1_flat, gradOutput)
  self.gradInput[2] = self.buffer
  return self.gradInput
end

local Sign, parent = torch.class('nn.Sign', 'nn.Module')

function Sign:__init(normalize)
  parent.__init(self)
end


function Sign:updateOutput(input)
  self.output:resizeAs(input)
  torch.sign(self.output, input)

  return self.output
end


function Sign:updateGradInput(input, gradOutput)
   self.gradInput:resizeAs(gradOutput):copy(gradOutput)
   self.gradInput:zero()
   return self.gradInput
end

local SignedSquareRoot, parent = torch.class('nn.SignedSquareRoot', 'nn.Module')

function SignedSquareRoot:__init(args)
   parent.__init(self)
   self.module = nn.Sequential()
      :add(nn.Abs())
      :add(nn.Sqrt())
end

function SignedSquareRoot:updateOutput(input)
   self.output = self.module:forward(input)
   self.tmp = self.tmp or input.new()
   self.tmp:resizeAs(input)
   torch.sign(self.tmp, input)
   self.output:cmul(self.tmp)
   return self.output
end

function SignedSquareRoot:updateGradInput(input, gradOutput)
   local eps = 1e-1  -- to avoid gradient explosion
   torch.cmul(self.gradInput, gradOutput, 
      torch.pow(self.module:forward(input)+eps,-1)/2)
   return self.gradInput
end