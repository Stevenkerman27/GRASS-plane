import torch
import util
from grassmodel import GRASSEncoder
from train_GAN import GANDiscriminator

def test_discriminator():
    config = util.get_args()
    config.cuda = False
    encoder = GRASSEncoder(config)
    discriminator = GANDiscriminator(encoder, config)
    
    # Dummy feature vector (output of encoder)
    dummy_feature = torch.randn(1, config.feature_size)
    
    score = discriminator.fc(dummy_feature)
    assert score.shape == (1, 1)
    print("test_discriminator passed")

if __name__ == "__main__":
    test_discriminator()
