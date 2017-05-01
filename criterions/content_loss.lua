local ContentLoss, parent = torch.class('nn.ContentLoss', 'nn.Module')


-- Assumes two passes, first sets target, second computes with reconstruction
-- Necessary for invariant autoencoder

function ContentLoss:__init(strength, normalize, name)
  parent.__init(self)
  self.strength = strength
  self.normalize = normalize or false
  self.name = name

  self.loss = 0
  self.target = nil
  self.output = nil

  self.criterion = nn.MSECriterion()
end

function ContentLoss:updateOutput(input)
  -- first pass means setting target
  if self.target == nil then
    -- print('Set target!')
    self.target = input:clone()
  else
    -- print('Computed target!')
    self.loss = self.criterion:forward(input, self.target)
  end

  self.output = input

  return self.output
end

function ContentLoss:updateGradInput(input, gradOutput)
  assert(input:nElement() == self.target:nElement(), 'Wrong input and target sizes!')

  self.gradInput = self.criterion:backward(input, self.target)

  if self.normalize then
    self.gradInput:div(torch.norm(self.gradInput, 1) + 1e-8)
  end

  self.gradInput:mul(self.strength)
  if gradOutput ~= nil then
     self.gradInput:add(gradOutput)
  end
  
  -- resets target
  self.target = nil
  return self.gradInput
end
