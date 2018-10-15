import numpy as np

predicted = {'classes': 7,
             'probabilities': np.array([1.9553413e-13, 1.0163559e-10, 2.1170275e-10, 8.1893268e-06,
                                        1.6208340e-09, 2.7085177e-08, 4.8949983e-15, 9.9999142e-01,
                                        1.4859140e-10, 3.8220912e-07], dtype=np.float32)}
prob = list(predicted['probabilities'])
print(prob.index(max(prob)))
