from Paths import *
from SALGAN.Our_SALGAN import ModelSALGAN
import test



def main():
    model = ModelSALGAN(batch_size=32)
    test.load_weights(model.net['output'], path='gen_', epochtoload=90)
    test.make_test(path_to_images='/home/lior/Desktop/Project_Deep_Learning/images', path_output_maps='/home/lior/Desktop/', model_to_test=model)

if __name__ == "__main__":
    main()
