from config import opt
import torch as t
import models

model = getattr(models, opt.model)()
if opt.load_model_path:
    checkpoint = t.load(opt.load_model_path)
    model.load_state_dict(checkpoint["state_dict"])
t.save(model.state_dict(), '/opt/checkpoint/' + opt.model + '.pth')
