from forecasting_models.pytorch.tools_2 import *
from forecasting_models.pytorch.models import *

model, _ = make_model('TransformerNet', 128, 128, None, 0.03, 'relu', 10, 'classfification', torch.device('cpu'), 3, 5, None)

for name, layer in model.named_children():
    print(layer.out_features)