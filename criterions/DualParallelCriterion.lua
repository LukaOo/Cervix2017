--[[
   Special criterion to aggregate table input [2:] into second criterion 
]] 

local DualParallelCriterion, parent = torch.class('nn.DualParallelCriterion', 'nn.Criterion')

function DualParallelCriterion:__init(repeatTarget)
   parent.__init(self)
   self.criterions = {}
   self.weights = {}
   self.gradInput = {}
   self.narrowInput = nn.NarrowTable(2,2) -- get all elements except first
end

function DualParallelCriterion:add(criterion, weight)
   assert(criterion, 'no criterion provided')
   assert(#self.criterions <= 2, 'count criterion can not be more then 2')
   weight = weight or 1
   table.insert(self.criterions, criterion)
   table.insert(self.weights, weight)
   return self
end

function DualParallelCriterion:updateOutput(input, target)
  
   self.output = self.weights[1]*self.criterions[1]:updateOutput(input[1],target)
   local output2 = self.criterions[2]:updateOutput(self.narrowInput(input),target)
   self.criterions[2].output = output2
   self.output = self.output + self.weights[2]*output2
   
   return self.output
end

function DualParallelCriterion:updateGradInput(input, target)
   self.gradInput     = nn.utils.recursiveResizeAs(self.gradInput, input)
   self.gradInput[1]  = self.weights[1] * self.criterions[1]:updateGradInput(input[1], target)
   local ograd        = self.criterions[2]:updateGradInput(self.narrowInput(input), target)
   self.gradInput[2]  = self.weights[2] * ograd[1]
   self.gradInput[3]  = self.weights[2] * ograd[2]
   
   return self.gradInput
end


function DualParallelCriterion:type(type, tensorCache)
   self.gradInput = {}
   return parent.type(self, type, tensorCache)
end