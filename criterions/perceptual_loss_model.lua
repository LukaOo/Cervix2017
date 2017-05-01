require 'nn'
require 'cutorch'
require 'criterions/content_loss'
require 'loadcaffe'
require 'models/decoder_module'

local PerceptualLossCriterion, parrent = torch.class('nn.PerceptualLossCriterion', 'nn.Criterion')

--
-- config
-- model_prototxt - prototxt file name
-- model model - file name
-- layer_name  - name of layer for perceptual loss
--
function PerceptualLossCriterion:__init( config, batchSize )
      assert(config ~= nil)
      self.batchSize = batchSize
      self.loss_modules = {}
      self.normalize = config.normalize_grad
      self.calc_only_target = config.calc_only_target

      local img_size = config.img_size or 224
      print ('Loading model: ', config.model_prototxt, config.model_file, 'features from: ', config.layer_name)

      model = loadcaffe.load(config.model_prototxt, config.model_file, 'nn')
      self.loss_module = self:__extract_features(model, config.layer_name, img_size)
      print ('Loss module: ', self.loss_module)
      self.crit = nn.MSECriterion()

      self.loss_module:evaluate()
      
      if config.gpu ~= nil and cutorch.getDevice() ~= config.gpu then
         print ('Perceptual loss gpu: ', config.gpu)
         self.gpu = config.gpu
      end
end

function tablelength(T)
  local count = 0
  for _ in pairs(T) do count = count + 1 end
  return count
end

function PerceptualLossCriterion:__extract_features(model, name, img_size)
     local loss_module = nn.Sequential()
     local l_name = name
     local c_modules = tablelength(l_name)

     --loss_module:add(CreateResnetOutputDeprocess(img_size))
     loss_module:add(CreateVGGInputPreprocessor(103.939, 116.779, 123.68, img_size))

     local lm = nn.ContentLoss( 1.0, false, 'MSE' .. '_content' )
     loss_module:add(lm)
     self.loss_modules['MSE'] = lm

     for i, m in ipairs(model:listModules()) do
        if i > 1 then
          loss_module:add(m)
          if l_name[ m.name ] ~=nil then
             local lm = nn.ContentLoss(l_name[ m.name ], false , m.name .. '_content' )
             loss_module:add( lm )
             self.loss_modules[m.name] = lm

             print (c_modules, 'Add perseptual loss: ' .. m.name .. ' strength: ' .. l_name[ m.name ] )
             if c_modules == 1  then
                return loss_module
             end
             c_modules = c_modules - 1        
          end
        end
     end
     error ('Can not find module: ', name)
end

local function switch_gpu(gpu)
   if gpu == nil then return nil end
   local cur_gpu = cutorch.getDevice()
   if cur_gpu ~= gpu then cutorch.setDevice(gpu) end
--   print ("Switch gpu: ", gpu)
   return cur_gpu
end
--

function PerceptualLossCriterion:updateOutput(input, target)
     local t = target
     local i = input
     if self.calc_only_target == true then
     -- first is target domain
     -- 
        t = t:narrow(1,1, self.batchSize)
        i = i:narrow(1,1, self.batchSize)
     end
     local cur_gpu = switch_gpu(self.gpu)
     if self.gpu ~= nil then 
        t = t:clone()
        i = i:clone()
     end
     for n, m in pairs(self.loss_modules) do
        m.target = nil
     end

     local loss_target = self.loss_module:forward(t)
     local loss_input  = self.loss_module:forward(i)

     self.output = 0 
     for n, m in pairs(self.loss_modules) do
        self.output = self.output + m.loss
     end

     switch_gpu(cur_gpu)

     -- print ("Cont loss:", self.output)
     return self.output
end

function PerceptualLossCriterion:updateGradInput(input, target)
    local t = target
    local i = input
   
    local cur_gpu = switch_gpu(self.gpu)
    if self.gpu ~= nil then 
        t = t:clone()
        i = i:clone()
    end
    -- reset targets for each module
    for n, m in pairs(self.loss_modules) do
        m.target = nil
    end

    local loss_target = self.loss_module:forward(t) -- store target output
    local loss_input  = self.loss_module:forward(i)

    self.gradInput = self.loss_module:updateGradInput(i, nil)

    if self.normalize == true then
       self.gradInput:div(torch.norm(self.gradInput, 1) + 1e-8)
    end

    if self.calc_only_target == true then
     -- first is target domain
     -- 
     -- zerroing gradient for source domain
       self.gradInput:narrow(1, self.batchSize + 1, self.batchSize):zero()
    end
    
    switch_gpu(cur_gpu)
    
    if self.gpu ~= nil then self.gradInput = self.gradInput:clone() end
    
    -- print ("Min:", torch.min(self.gradInput), "Max: ", torch.max(self.gradInput), "Mean: ", torch.mean(self.gradInput))
    return self.gradInput   
end

-- return nn.PerceptualLossCriterion(perceptual_config, opt.batchSize)
