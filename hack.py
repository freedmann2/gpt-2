import tensorflow as tf


# SETUP INSTRUCTIONS: first, run `python3 download_model.py 117M`

# STEP TWO: there is no step two.


ckpt = tf.train.load_checkpoint('models/117M/model.ckpt')
import pdb; pdb.set_trace()


wpe = ckpt.get_tensor('model/wpe')
wte = ckpt.get_tensor('model/wte')

# e.g.



# (Pdb) wpe = ckpt.get_tensor('model/wpe')
# (Pdb) wpe.shape
# (1024, 768)
# (Pdb) wte = ckpt.get_tensor('model/wte')
# (Pdb) wte.shape
# (50257, 768)




# at this point, `wpe` looks like:


# array([[-1.8820720e-02, -1.9741860e-01,  4.0267250e-03, ...,
#         -4.3043736e-02,  2.8267192e-02,  5.4490108e-02],
#        [ 2.3959434e-02, -5.3792033e-02, -9.4878644e-02, ...,
#          3.4170013e-02,  1.0171850e-02, -1.5572949e-04],
#        [ 4.2160717e-03, -8.4763914e-02,  5.4514930e-02, ...,
#          1.9744711e-02,  1.9324856e-02, -2.1423856e-02],
#        ...,
#        [-1.7986511e-03,  1.6052092e-03, -5.5103153e-02, ...,
#          1.3616630e-02, -7.1805264e-03,  3.7552188e-03],
#        [ 3.2105497e-03,  1.5500595e-03, -4.8944373e-02, ...,
#          2.0725457e-02, -1.1837787e-02, -5.5682898e-04],
#        [ 2.6609693e-04,  3.0272407e-03, -1.7086461e-03, ...,
#         -4.6505518e-03, -2.3541194e-03, -5.7855090e-03]], dtype=float32)
