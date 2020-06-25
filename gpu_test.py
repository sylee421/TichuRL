# import os
# import tensorflow as tf

# os.environ['CUDA_VISIBLE_DEVICES'] = '0'    

# if tf.test.gpu_device_name():
#     print('GPU found')
# else:
#     print("No GPU found")

# GPU found


import os
print("os import")
import tensorflow as tf
print("tensorflow import")
os.environ['CUDA_VISIBLE_DEVICES'] = '0' # 선택한 gpu에만 메모리 할당
print("os cuda device specify")

if tf.test.gpu_device_name():
    print('GPU found')
else:
    print("No GPU found")