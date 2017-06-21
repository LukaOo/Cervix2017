require 'xlua'
require 'optim'
require 'nn'
require 'stn'
require 'dpnn'
require 'models/gradient_decrease'
local utils=require 'utils'

-- dofile './provider.lua'
local c = require 'trepl.colorize'

opt = lapp[[
   -i,--input                 (default "")          input data file
   -s,--save                  (default "logs")      subdirectory to save logs
   -b,--batchSize             (default 128)         batch size
   -r,--learningRate          (default 1)           learning rate
   --learningRateDecay        (default 1e-7)        learning rate decay
   --lr_decay_sheduler        (default {})          learning rate decay sheduler {[10]=0.5}
   --weightDecay              (default 0.0005)      weightDecay
   -m,--momentum              (default 0.9)         momentum
   --epoch_step               (default 25)          epoch step
   --model                    (default vgg_bn_drop)     model name
   --max_epoch                (default 300)           maximum number of iterations
   --backend                  (default nn)            backend
   --type                     (default cuda)          cuda/float/cl
   --net_config               (default "{cinput_planes=240, image_size=256 }")         net configuration 
   --provider_config          (default "{}" )      provider configuration
   --optim                    (default "sgd")      optimizer
   --use_optnet               (default 1)          use memory optimisation by optnet
   --checkpoint               (default '')         use path for checkpoints
   --continue                 (default '')         use model to continue learning
   --save_epoch               (default 10)         save checkpoints of neural net each N epoch
   --criterion                (default 'CrossEntropy') criterion CrossEntropy|Dice|SpatialBCE|MCE|PL
   --perceptual_config        (default nil)        perceptual criterion configuration
   --ignore_state             (default 0 )         ignore states when load model to continie training
   --crit_config              (default {})         criterion configuration {weights={0.5,1}} 
   --grad_noise               (default nil)        gradient noise regularization {var=0.1}
]]

print(opt)

function file_exists(name)
   local f=io.open(name,"r")
   if f~=nil then io.close(f) return true else return false end
end


provider_config = loadstring(" return " .. opt.provider_config) ()
net_config = loadstring(" return " .. opt.net_config)()
lr_decay_sheduler = loadstring(" return " .. opt.lr_decay_sheduler)()
crit_config = loadstring(" return " .. opt.crit_config)()
grad_noise  = loadstring(" return " .. opt.grad_noise)()

local function cast(t)
   if opt.type == 'cuda' then
      require 'cunn'
      return t:cuda()
   elseif opt.type == 'float' then
      return t:float()
   elseif opt.type == 'cl' then
      require 'clnn'
      return t:cl()
   else
      error('Unknown type '..opt.type)
   end
end

if opt.input == '' or file_exists(opt.input) == false then
     error("Can not open file: '"..opt.input.."'")
end

local sprovider   = provider_config.provider..'.lua'
dofile (sprovider)

print(c.blue '==>' ..' configuring model')
local model = nil 



if opt.continue == '' then
model = cast(dofile('models/'..opt.model..'.lua'))

--utils.InitNetwork(model, true)

if opt.use_optnet == 1 then
   optnet = require 'optnet'
   print ("Count used memory before: ", optnet.countUsedMemory(model))
   if provider_config.volumetric == true then
     mod_input = cast(torch.rand(2, 1, 1, 32,32,32))
   else
      if provider_config.flat == true then
          mod_input = cast(torch.rand( 2, net_config.cinput_planes, net_config.input_size))
      else
          if provider_config.siames_input == true then
            local tt = {}
            mod_input = cast(torch.rand( 1, net_config.cinput_planes, net_config.image_size, net_config.image_size))
            tt[1] = mod_input
            tt[2] = mod_input:clone()
            if provider_config.triplets == true then tt[3] = mod_input:clone() end
            mod_input = tt
          else
            mod_input = cast(torch.rand( 1, net_config.cinput_planes, net_config.image_size, net_config.image_size))
          end
      end
   end
   mod_opts = {inplace=false, mode='training'}
   optnet.optimizeMemory(model, mod_input, mod_opts)
   print ("Count used memory after: ", optnet.countUsedMemory(model))
end

else
   print (c.cyan'Loading model', opt.continue)
   model = cast(torch.load(opt.continue))
end

if opt.backend == 'cudnn' then
   require 'cudnn'
   print ('Converting to cudnn')
   cudnn.convert(model, cudnn)
end
gpus = torch.range(1, cutorch.getDeviceCount()):totable()
print ('Gpus', gpus)

if #gpus > 1 then
  dpt = nn.DataParallelTable(1):add(model, gpus):cuda()
else
  dpt = model
end


print(model)
if grad_noise ~= nil then
  print ('Gardient noise regularisation with start variance - ', grad_noise.var)
end
print(c.blue '==>' ..' loading data ' .. opt.input)

provider_config.input_path = opt.input

provider_config.data_set_name = 'train'
provider_config.model_type = opt.type

trainData = Hdf5Provider( provider_config, opt.batchSize )

provider_config.data_set_name = 'test'
testData  = Hdf5Provider( provider_config, opt.batchSize )

confusion = opt.criterion == 'CrossEntropy' and optim.ConfusionMatrix( net_config.class_count ) or nil

print('Will save at '..opt.save)

paths.mkdir(opt.save)
testLogger = optim.Logger(paths.concat(opt.save, 'test.log'))
if confusion == nil then
   if opt.criterion ~= 'Dual' and (crit_config.cross_entropy ~= true and opt.criterion ~= 'DistanceRatio') then
      testLogger:setNames{opt.criterion.. ' loss (train set)', opt.criterion..' loss (test set)'}
   else
      testLogger:setNames{opt.criterion.. ' loss (train set)', opt.criterion..' loss (test set)', 'CE loss (train set)', 'CE loss (test set)', 'HE loss (train set)', 'HE loss (test set)'}
   end
else  
   testLogger:setNames{'Mean class accuracy (train set)', 'Mean class accuracy (test set)', 'Train error', 'Test error'}
end
testLogger.showPlot = false

parameters,gradParameters = dpt:getParameters()

print (parameters:size(), gradParameters:size())

print(c.blue'==>' ..' setting criterion ' .. opt.criterion)
if opt.criterion == 'CrossEntropy' then
   criterion = cast(nn.CrossEntropyCriterion())
else
  if opt.criterion == 'Dice' then
     require 'criterions/dice_coeff_loss'
     criterion = cast( nn.DICECriterion())
  else
     if opt.criterion == 'SpatialBCE' then
        require 'criterions/SpatialBCECriterion'
        criterion = cast( nn.SpatialBCECriterion()) --nn.DICECriterion())
     else
       if opt.criterion == 'MSE' then
          criterion = cast( nn.MSECriterion())
       else
          if opt.criterion == 'PL' then
            require 'criterions/perceptual_loss_model'
            perceptual_config = loadstring(" return " .. opt.perceptual_config) ()
            criterion = cast( nn.PerceptualLossCriterion(perceptual_config, opt.batchSize) )
          else
            if opt.criterion == 'BCE' then
              criterion = cast( nn.BCECriterion() )
            else
               if opt.criterion == 'Cosine' then
                  criterion = cast( nn.CosineEmbeddingCriterion() )
                else
                   if opt.criterion == 'Dual' then
                     local w = {1,1}
                     if crit_config ~= nil and crit_config.weights ~= nil and #crit_config.weights == 2 then
                       w = crit_config.weights
                     end
                     print ("Dual criterion weights: ", w)
                     criterion = cast( nn.ParallelCriterion():add(nn.CrossEntropyCriterion(), w[1]):add(nn.HingeEmbeddingCriterion(), w[2]) )
                   else
                     if opt.criterion == 'DistanceRatio' then
                         if crit_config.cross_entropy == true then
                           require 'criterions/DualParallelCriterion'
                           criterion = cast( nn.DualParallelCriterion():add(nn.CrossEntropyCriterion()):add(nn.DistanceRatioCriterion()) )
                         else 
                           criterion = cast( nn.DistanceRatioCriterion() )
                         end
                     end
                   end
               end
            end   
          end   
       end
     end
  end
end




print(c.blue'==>' ..' configuring optimizer')
if opt.continue == '' or opt.ignore_state == 1 then
optimState = {
  learningRate = opt.learningRate,
  weightDecay = opt.weightDecay,
  momentum = opt.momentum,
  learningRateDecay = opt.learningRateDecay,
}

else
  print ('Loading state: ',  opt.continue .. '.ostat')
  optimState = torch.load(opt.continue .. '.ostat')
end

last_error = 99
if model.last_error ~= nil then
  last_error = model.last_error
end

local function get_step(optstate)
   local step = optstate.t or optstate.evalCounter
   step  = step or 0
   return step
end
local function get_lr(optstate)
   local step = get_step(optstate)
   local lrd = optstate.learningRateDecay
   local lr  = optstate.learningRate 
   local clr = lr / (1 + step*lrd)   
   return clr
end

local function cast_target(inputs)
  
    if type(inputs) == 'table' then
       for ii=1, #inputs do
          inputs[ii] = cast(inputs[ii])
       end
    else
          inputs  = cast(inputs)
    end
    
    return inputs
end

local function get_dual_loss_from(crit, output_loss)
    if opt.criterion == 'Dual' or (crit_config.cross_entropy == true and opt.criterion == 'DistanceRatio')then
      if ( torch.typename(crit) == 'nn.ParallelCriterion' or torch.typename(crit) == 'nn.DualParallelCriterion')  and output_loss ~= nil then
        output_loss[1] = output_loss[1] + crit.criterions[1].output 
        output_loss[2] = output_loss[2] + crit.criterions[2].output 
      end
    else
       output_loss = nil
    end
    return output_loss
end

function train()
  dpt:training()
  epoch = epoch or 1

  -- drop learning rate every "epoch_step" epochs
  --if epoch % opt.epoch_step == 0 then optimState.learningRate = optimState.learningRate/2 end
  if lr_decay_sheduler ~= nil and lr_decay_sheduler[epoch] ~= nil then
    optimState.learningRate = optimState.learningRate * lr_decay_sheduler[epoch]
  end
  
  print(c.blue '==>'.." online epoch # " .. epoch .. ' [batchSize = ' .. opt.batchSize .. ']')
  print(c.green " Epoch learning rate: ", optimState.learningRate, get_lr( optimState ), ' Real Iteration ' , get_step(optimState) )
  
  local tic = torch.tic()
  local iters = opt.epoch_step <= 0 and trainData.cbatches or math.min(trainData.cbatches, opt.epoch_step )
  trainData:reset()
  local sum_error = 0
  local dual_loss = {}
  dual_loss[1] = 0
  dual_loss[2] = 0
  
  for t=1, iters do
    xlua.progress(t, iters)

    local inputs, targets = trainData:get_next_batch()
	

    inputs = cast_target(inputs)
    targets = cast_target(targets) 
	

    local feval = function(x)
      if x ~= parameters then parameters:copy(x) end
      dpt:zeroGradParameters()
      
      local outputs = dpt:forward(inputs)
      
	  	if opt.checkpoint ~= nil and opt.checkpoint ~= '' 
        and ( (optimState.evalCounter~= nil and optimState.evalCounter % 50 == 0) or (optimState.t ~= nil and optimState.t % 50 == 0) )  then
        local it = optimState.evalCounter or optimState.t
        if type(inputs) == 'table' then
          for ii=1, #inputs do
            torch.save(opt.checkpoint .. '/check_point.' .. ii .. '.' .. it, inputs[ii]:float())
          end
        else
         torch.save(opt.checkpoint .. '/check_point.' .. it, inputs:float())
        end
        
        torch.save(opt.checkpoint .. '/check_point_t.' .. it, targets:float())
        if type(outputs) == 'table' then
          for ii=1, #inputs do
            torch.save(opt.checkpoint .. '/check_point_o.' .. ii .. '.' .. it, outputs[ii]:float())
          end
        else
            torch.save(opt.checkpoint .. '/check_point_o.' .. it, outputs:float())
        end
      end

      local f = criterion:forward(outputs, targets)
      
      dual_loss = get_dual_loss_from(criterion, dual_loss)
      
      if opt.optim ~= 'check' then 
        print (f) 
      else
        xlua.progress(check_it, gradParameters:size(1) * 2)
        check_it = check_it + 1
      end
      sum_error = sum_error + f
      
      local df_do = criterion:backward(outputs, targets)
      dpt:backward(inputs, df_do)
      
      if confusion then confusion:batchAdd( outputs, targets) end
      --  gradient noise regulariazation
      if grad_noise ~= nil then
         if not optimState.noiseParameters then
            optimState.noiseParameters = torch.Tensor():typeAs(x):resizeAs(gradParameters)
         end
         local noise_t = get_step(optimState)
         gradParameters:add(optimState.noiseParameters:normal(0, grad_noise.var / math.pow ((1+noise_t), 0.55)))
      end
      return f,gradParameters
    end
    
    if opt.optim == 'adam' then
       optim.adam(feval, parameters, optimState)
    else
	  if opt.optim == 'check' then
            check_it = 1
	    local diff, dC, dC_est = optim.checkgrad(feval, parameters)
            print ("Diff:", diff, torch.norm(dC-dC_est), torch.norm(dC+dC_est), torch.max(dC-dC_est))
	  else
             if opt.optim == 'rmsprop' then
                optim.rmsprop(feval, parameters, optimState)
             else
                optim.sgd(feval, parameters, optimState)
             end
	  end
    end
    
  end

  if confusion then
    confusion:updateValids()
    print(('Train accuracy: '..c.cyan'%.2f'..' %%\t time: %.2f s, avg error: %.4f'):format(
            confusion.totalValid * 100, torch.toc(tic), sum_error / iters ))
        
    print ('Confusion matrix: ', confusion)
    train_acc   = confusion.totalValid
    confusion:zero()
  else
    print (("Train error: ".. c.cyan'%.4f' .. ' %%\t time: %.2f s' ):format( sum_error / iters, torch.toc(tic)))
  end
  if dual_loss ~= nil then
     dual_loss[1] =  dual_loss[1] / iters
     dual_loss[2] =  dual_loss[2] / iters
     print (("Train CE error: ".. c.cyan'%.4f' .. "; HE error: ".. c.cyan'%.4f' .. ' %%\t time: %.2f s' ):format( dual_loss[1], dual_loss[2], torch.toc(tic)) )
  end

  train_error     = sum_error / iters
  train_dual_loss = dual_loss

  epoch = epoch + 1
end


function test()
  -- disable flips, dropouts and batch normalization
  dpt:evaluate()
  print(c.blue '==>'.." testing")
  local bs = opt.batchSize
  local sum_error = 0
  local iters = 0
  local cc  = math.ceil(testData:size()/bs)
  local dual_loss = {}
  dual_loss[1] = 0
  dual_loss[2] = 0

  
  testData:reset()
  for i=1, cc do
    xlua.progress(i,  cc)
    
    local k = (i-1)*bs + 1
    
    local inputs, targets  = testData:sub(k,k+bs-1)

    inputs  = cast_target(inputs)
    targets = cast_target(targets)
    
    local outputs = dpt:forward(inputs)
    local e = criterion:forward(outputs, targets)
    
    dual_loss = get_dual_loss_from(criterion, dual_loss)
    
    sum_error = sum_error + e
    iters = iters + 1
    
    if confusion then confusion:batchAdd( outputs,targets) end
  end

  if confusion then
    confusion:updateValids()
    print('Test accuracy:', confusion.totalValid * 100, " Avg test err: ", sum_error / iters )
    print ('Confusion matrix: ', confusion)
  else
    print (("Test error: ".. c.cyan'%.4f' ):format( sum_error / iters ))
  end
  
  if dual_loss ~= nil then
     dual_loss[1] =  dual_loss[1] / iters
     dual_loss[2] =  dual_loss[2] / iters
     print (("Test CE error: ".. c.cyan'%.4f' .. "; HE error: ".. c.cyan'%.4f' ):format( dual_loss[1], dual_loss[2]) )
  end

  test_error = sum_error / iters
  test_dual_loss = dual_loss

  if testLogger then
    paths.mkdir(opt.save)
    if confusion then
      testLogger:add{train_acc, confusion.totalValid, train_error, test_error}
      testLogger:style{'-','-', '-', '-' }
    else
      if dual_loss == nil then
        testLogger:add{train_error, test_error}
        testLogger:style{'-','-', '-', '-' }
      else
        testLogger:add{train_error, test_error, train_dual_loss[1], test_dual_loss[1], train_dual_loss[2], test_dual_loss[2]}
        testLogger:style{'-','-', '-', '-', '-', '-' }
      end
    end  
    testLogger:plot()

    if paths.filep(opt.save..'/test.log.eps') then
      local base64im
      do
        os.execute(('convert -density 200 %s/test.log.eps %s/test.png'):format(opt.save,opt.save))
        os.execute(('openssl base64 -in %s/test.png -out %s/test.base64'):format(opt.save,opt.save))
        local f = io.open(opt.save..'/test.base64')
        if f then base64im = f:read'*all' end
      end

      local file = io.open(opt.save..'/report.html','w')
      file:write(([[
      <!DOCTYPE html>
      <html>
      <body>
      <title>%s - %s</title>
      <img src="data:image/png;base64,%s">
      <h4>optimState:</h4>
      <table>
      ]]):format(opt.save,epoch,base64im))
      for k,v in pairs(optimState) do
        if torch.type(v) == 'number' then
          file:write('<tr><td>'..k..'</td><td>'..v..'</td></tr>\n')
        end
      end
      file:write'</table><pre>\n'
      file:write(tostring(confusion)..'\n')
      file:write(tostring(model)..'\n')
      file:write'</pre></body></html>'
      file:close()
    end
  end

  -- save model every 50 epochs
  if test_error  < last_error then
    local filename = paths.concat(opt.save, 'model.t7')
    model.last_error = test_error
    print('==> saving model to '..filename)
    torch.save(filename, model:clearState())
    torch.save(filename .. '.stat', optimState)
--    if last_error ~= 0 then
--       optimState.learningRate = optimState.learningRate * 0.9
--    end
    last_error  = test_error
  else
     if (epoch % opt.save_epoch) == 0 then
        local filename = paths.concat(opt.save, 'checkpoint.t7')
        print('==> saving model checkpoint to '.. filename)
        torch.save(filename, model )
        torch.save(filename .. '.ostat', optimState )
     end    
  end

  if confusion then confusion:zero() end
end


for i=1,opt.max_epoch do
  train()
  test()
end


