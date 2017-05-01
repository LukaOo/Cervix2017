require 'nn'
require 'cunn'
require 'models/gradient_decrease'
-- require 'models/gradient_debug'
-- require 'models/decoder_module'

cudnn = require 'cudnn'

local MODEL_FILE = './pretrained/'.. net_config.model_file
local class_count = net_config.class_count
local gradient_decrease = net_config.gradiend_decrease or 0.1
local fc_dropout        = net_config.fc_dropout or 0.3
local fc_conf           = net_config.fc

local resnet = torch.load(MODEL_FILE)
local linear_input_size = 0

for i,lineria in ipairs(resnet:findModules('nn.Linear'))  do
    lineria.name = 'Classifier'
    linear_input_size = lineria.weight:size(2)
    print (lineria)
end


local last_layer_size = linear_input_size

-- 
-- Add final classificator
--
local classifier = nn.Sequential()
      classifier:add(nn.GradientDecrease(gradient_decrease))
--      classifier:add(nn.Dropout(0.5))
if fc_conf == nil then
      classifier:add(nn.Linear( linear_input_size, last_layer_size))
      classifier:add(nn.BatchNormalization(last_layer_size)) 
      classifier:add(nn.ReLU(true))
      classifier:add(nn.Dropout(fc_dropout))
else
  inputsize = linear_input_size
  for i=1, #fc_conf do
     print ("Fc layer: ", i, fc_conf[i])
     classifier:add(nn.Linear( inputsize, fc_conf[i].size))
     if fc_conf[i].bn == true then
       classifier:add(nn.BatchNormalization(fc_conf[i].size)) 
     end
     if fc_conf[i].relu == true then
       classifier:add(nn.ReLU(true)) 
     end
     if fc_conf[i].lrelu ~= nil then
       classifier:add(nn.LeakyReLU(fc_conf[i].lrelu, true))
     end
     if fc_conf[i].dropout ~= nil then
       classifier:add(nn.Dropout(fc_conf[i].dropout))
     end
     inputsize = fc_conf[i].size
  end
  last_layer_size = inputsize
end
      classifier:add(nn.Linear(last_layer_size, class_count ) )


resnet = resnet:replace(function(module)
         if torch.typename(module) == 'nn.Linear' and module.name == 'Classifier' then
              return classifier
         else
              return module
           end
         end)


return resnet
