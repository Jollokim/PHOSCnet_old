import models

from timm import create_model
from torchsummary import summary

model = create_model('PHOSCnet_temporalpooling')

summary(model, (3, 50, 250))

