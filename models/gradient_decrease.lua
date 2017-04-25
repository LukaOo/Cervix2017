local GradientDecrease, parent = torch.class('nn.GradientDecrease', 'nn.Module')

function GradientDecrease:__init(strength)
  parent.__init(self)
  self.strength = strength
end

function GradientDecrease:updateOutput(input)
  self.output = input
  self.size = input:size()
  return self.output
end

function GradientDecrease:updateGradInput(input, gradOutput)
  self.gradInput:resizeAs(gradOutput):copy(gradOutput)
  self.gradInput:mul(self.strength)
  return self.gradInput
end

