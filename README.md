# Temporal PHOSCnet

This is a PHOSCnet version for recognizing handwritten words in images based on the PHOSCnet from the paper: [Pho(SC)-CTC - A Hybrid Approach Towards Zero-shot Word
Image Recognition](https://arxiv.org/pdf/2105.15093.pdf). Please cite this paper when using this repository for future work.

The major difference between this Temporal PHOSCnet and the original PHOSCnet lies within the transition between the Convolutional layers and the regular neural net layers.
Instead of using a 3-level Spatial Pyramid Max Pooling layer, the layer is switched out with a 3-level Temporal Pyramid Pooling Layer. The two Pyramid pooling layers 
both have the job of making sure feature maps from the convolutional layers fits the input channel of the first MLP layer. However, the method which they do is different.

[Spatial Pyramid Pooling](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=7005506&casa_token=UuRaEPYYiyAAAAAA:4pcm6cp4eaGjwsTuKnB-outFHSb5n2n0yYkTTuqTwQpPxOtdnbX8cFbh8P2VLBaCiWOgg2hHSZHL)

[Temporal Pyramid Pooling](https://patrec.cs.tu-dortmund.de/pubs/papers/Sudholt2017-EWS.pdf)

## Results
Training and testing with the IAM dataset.

seen accuracy: 92%

unseen accuracy: 82%
