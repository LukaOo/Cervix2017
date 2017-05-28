require 'nn'


local inputsize = net_config.inputsize
local classifier = nn.Sequential()
local class_count = net_config.class_count
local fc_conf = net_config.fc

if fc_conf == nil then
      last_layer_size = inputsize
      classifier:add(nn.Linear( inputsize, last_layer_size))
      classifier:add(nn.BatchNormalization(last_layer_size)) 
      classifier:add(nn.ReLU(true))
      classifier:add(nn.Dropout(fc_dropout))
else
  
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

-- add final layer
classifier:add(nn.Linear(last_layer_size, class_count ) )

return classifier