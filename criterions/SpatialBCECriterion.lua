--[[
      Implements Spatial Binary Cross Entropy loss
      -y*log(x)-(1-y)log(1-x) - where x is a pixel of output image

]]

local SpatialBCECriterion, parent = torch.class('nn.SpatialBCECriterion', 'nn.Criterion')

local eps = 0.001

function SpatialBCECriterion:__init(w_one, w_zero, eps)
	parent.__init(self)
        self.eps = eps or 1e-6
        self.w_one = w_one or 1
        self.w_zero = w_zero or 1

end

function SpatialBCECriterion:updateOutput(input, i_target)
        if type(i_target) == 'table' then
          target = i_target[1]
        else
          target = i_target
        end
	      assert(input:nElement() == target:nElement(), "input and target size mismatch")
        assert(input:size(2) == 1,  "Input must have only single channel but has "..input:size(2))

	      local weights = self.weights

        local logx   = torch.log(input+self.eps)  * self.w_one
        local log1_x = torch.log(1 - input+self.eps) * self.w_zero
        local n      = input:size(1)   --  count samples in batch
        local h, w   = input:size(3),input:size(4)
        
        local output = torch.sum( -(torch.cmul(target,logx) + torch.cmul((1-target),log1_x)))
        output = output / (n*h*w)


	      self.output = output

	return self.output
end

--[[
    Grad output is -(y/x-(1-y)/(1-x))
]]
function SpatialBCECriterion:updateGradInput(input, i_target)
        if type(i_target) == 'table' then
          target = i_target[1]
        else
          target = i_target
        end

        local h, w   = input:size(3),input:size(4)

        assert(input:nElement() == target:nElement(), "inputs and target size mismatch")
        assert(input:size(2) == 1,  "Input must have only single channel but has "..input:size(2))

        local y_o_x     =  torch.cdiv( target, (input+self.eps)) * self.w_one
        local y_o_x_sub =  torch.cdiv( (1-target) , ((1-input)+self.eps)) * self.w_zero
        self.gradInput  =  ( y_o_x_sub - y_o_x) / (h*w)
--        print (torch.min(self.gradInput), torch.max(self.gradInput), torch.mean(self.gradInput))
        
        
	return self.gradInput
end

function SpatialBCECriterion:accGradParameters(input, gradOutput)
end

function SpatialBCECriterion:reset()
end
