
local DICECriterion, parent = torch.class('nn.DICECriterion', 'nn.Criterion')

local eps = 0.001

function DICECriterion:__init(weights)
	parent.__init(self)

	if weights then
	   assert(weights:dim() == 1, "weights input should be 1-D Tensor")
	   self.weights = weights
	end

end

function DICECriterion:updateOutput(input, target)

	assert(input:nElement() == target:nElement(), "input and target size mismatch")
	local weights = self.weights

	local numerator, denom, common, output

	-- compute 2 * (X intersection Y)
	numerator = torch.sum(torch.cmul( input, target) )  		--find logical equivalence between both
  numerator = numerator * 2 + eps

	-- compute denominator: sum_i(X) + sum_i(Y)
	denom = torch.sum(input) + torch.sum(target) + eps

	output = numerator/denom / input:size(1)

	self.output = -output

	return self.output
end

function DICECriterion:updateGradInput(input, target)
        ---
        ---
        ---
	assert(input:nElement() == target:nElement(), "inputs and target size mismatch")
	local denom = torch.sum(input) + torch.sum(target) + eps
	local gradInput = - 1/denom
	local part2     = 2 * ( target - ( torch.sum( torch.cmul( input, target) ) / denom ) )
        
        self.gradInput = gradInput * part2

        
	return self.gradInput
end

function DICECriterion:accGradParameters(input, gradOutput)
end

function DICECriterion:reset()
end
