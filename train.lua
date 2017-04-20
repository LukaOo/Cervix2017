require 'xlua'
require 'optim'
require 'nn'
require 'stn'
require 'dpnn'
local utils=require 'utils'

-- dofile './provider.lua'
local c = require 'trepl.colorize'

opt = lapp[[
   -i,--input                 (default "")          input data file
   -s,--save                  (default "logs")      subdirectory to save logs
   -b,--batchSize             (default 128)         batch size
   -r,--learningRate          (default 1)           learning rate
   --learningRateDecay        (default 1e-7)        learning rate decay
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
   --criterion                (default 'CrossEntropy') criterion CrossEntropy|Dice
]]

print(opt)

function file_exists(name)
   local f=io.open(name,"r")
   if f~=nil then io.close(f) return true else return false end
end


provider_config = loadstring(" return " .. opt.provider_config) ()
net_config = loadstring(" return " .. opt.net_config)()

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


if opt.backend == 'cudnn' then
   require 'cudnn'
   print ('Converting to cudnn')
   cudnn.convert(model, cudnn)
end

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
          mod_input = cast(torch.rand( 1, net_config.cinput_planes, net_config.image_size, net_config.image_size))
      end
   end
   mod_opts = {inplace=true, mode='training'}
   optnet.optimizeMemory(model, mod_input, mod_opts)
   print ("Count used memory after: ", optnet.countUsedMemory(model))
end

else
   print (c.cyan'Loading model')
   model = torch.load(opt.continue)
end
 
gpus = torch.range(1, cutorch.getDeviceCount()):totable()
print ('Gpus', gpus)

if #gpus > 1 then
  dpt = nn.DataParallelTable(1):add(model, gpus):cuda()
else
  dpt = model
end


print(model)

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
if opt.criterion == 'Dice' then
   testLogger:setNames{'Dice loss (train set)', 'Dice loss (test set)'}
else  
   testLogger:setNames{'Mean class accuracy (train set)', 'Mean class accuracy (test set)', 'Train error', 'Test error'}
end
testLogger.showPlot = false

parameters,gradParameters = dpt:getParameters()


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
     end
  end
end




print(c.blue'==>' ..' configuring optimizer')
if opt.continue == '' then
optimState = {
  learningRate = opt.learningRate,
  weightDecay = opt.weightDecay,
  momentum = opt.momentum,
  learningRateDecay = opt.learningRateDecay,
}

else
  optimState = torch.load(opt.continue .. '.ostat')
end

last_error = 99
if model.last_error ~= nil then
  last_error = model.last_error
end

function train()
  dpt:training()
  epoch = epoch or 1

  -- drop learning rate every "epoch_step" epochs
  --if epoch % opt.epoch_step == 0 then optimState.learningRate = optimState.learningRate/2 end
  
  print(c.blue '==>'.." online epoch # " .. epoch .. ' [batchSize = ' .. opt.batchSize .. ']')


  local tic = torch.tic()
  local iters = opt.epoch_step <= 0 and trainData.cbatches or math.min(trainData.cbatches, opt.epoch_step )
  trainData:reset()
  local sum_error = 0
  
  for t=1, iters do
    xlua.progress(t, iters)

    local inputs, targets = trainData:get_next_batch()
	

    

    inputs  = cast(inputs)
    targets = cast(targets) 
	

    local feval = function(x)
      if x ~= parameters then parameters:copy(x) end
      gradParameters:zero()
      
      local outputs = dpt:forward(inputs)
      
	  	if opt.checkpoint ~= nil and opt.checkpoint ~= '' and optimState.evalCounter~= nil and optimState.evalCounter % 50 == 0 then
        torch.save(opt.checkpoint .. '/check_point.' .. optimState.evalCounter, inputs:float())
        torch.save(opt.checkpoint .. '/check_point_t.' .. optimState.evalCounter, targets:float())
        torch.save(opt.checkpoint .. '/check_point_o.' .. optimState.evalCounter, outputs:float())
      end
      
      local f = criterion:forward(outputs, targets)
      print (f)
      sum_error = sum_error + f
      local df_do = criterion:backward(outputs, targets)
      dpt:backward(inputs, df_do)
      if confusion then confusion:batchAdd( outputs, targets) end

      return f,gradParameters
    end
    if opt.optim == 'adam' then
       optim.adam(feval, parameters, optimState)
    else
	  if opt.optim == 'check' then
	    optim.checkgrad(feval, parameters)
	  else
      optim.sgd(feval, parameters, optimState)
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

  train_error = sum_error / iters

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

  
  testData:reset()
  for i=1, cc do
    xlua.progress(i,  cc)
    
    local k = (i-1)*bs + 1
    
    local inputs, targets  = testData:sub(k,k+bs-1)

    inputs  = cast(inputs)
    targets = cast(targets)
    local outputs = dpt:forward(inputs)
    local e = criterion:forward(outputs, targets)
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
  test_error = sum_error / iters
  
  if testLogger then
    paths.mkdir(opt.save)
    if confusion then
      testLogger:add{train_acc, confusion.totalValid, train_error, test_error}
      testLogger:style{'-','-', '-', '-' }
    else
      testLogger:add{train_error, test_error}
      testLogger:style{'-','-', '-', '-' }
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


