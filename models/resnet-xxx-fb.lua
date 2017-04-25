require 'nn'
require 'cunn'
require 'models/gradient_decrease'
-- require 'models/gradient_debug'
-- require 'models/decoder_module'

cudnn = require 'cudnn'

local MODEL_FILE = './pretrained/'.. net_config.model_file
local class_count = net_config.class_count
local gradient_decrease = net_config.gradiend_decrease or 0.1 

local resnet = torch.load(MODEL_FILE)
local linear_input_size = 0

for i,lineria in ipairs(resnet:findModules('nn.Linear'))  do
    lineria.name = 'Classifier'
    linear_input_size = lineria.weight:size(2)
    print (lineria)
end


local last_layer_size = 512

-- 
-- Add final classificator
--
local classifier = nn.Sequential()
      classifier:add(nn.GradientDecrease(gradient_decrease))
--      classifier:add(nn.Dropout(0.5))
      classifier:add(nn.Linear( linear_input_size, last_layer_size))
      classifier:add(nn.BatchNormalization(last_layer_size)) 
      classifier:add(nn.ReLU(true))
      classifier:add(nn.Dropout(0.3))
      classifier:add(nn.Linear(last_layer_size, class_count ) )


resnet = resnet:replace(function(module)
         if torch.typename(module) == 'nn.Linear' and module.name == 'Classifier' then
              return classifier
         else
              return module
           end
         end)


return resnet
