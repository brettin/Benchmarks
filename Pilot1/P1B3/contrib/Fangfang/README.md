```
Input dim = 29281
____________________________________________________________________________________________________
Layer (type)                     Output Shape          Param #     Connected to
====================================================================================================
locallyconnected1d_1 (LocallyCon (None, 29281, 10)     585620      locallyconnected1d_input_1[0][0]
____________________________________________________________________________________________________
maxpooling1d_1 (MaxPooling1D)    (None, 292, 10)       0           locallyconnected1d_1[0][0]
____________________________________________________________________________________________________
flatten_1 (Flatten)              (None, 2920)          0           maxpooling1d_1[0][0]
____________________________________________________________________________________________________
dense_1 (Dense)                  (None, 1000)          2921000     flatten_1[0][0]
____________________________________________________________________________________________________
dropout_1 (Dropout)              (None, 1000)          0           dense_1[0][0]
____________________________________________________________________________________________________
dense_2 (Dense)                  (None, 500)           500500      dropout_1[0][0]
____________________________________________________________________________________________________
dropout_2 (Dropout)              (None, 500)           0           dense_2[0][0]
____________________________________________________________________________________________________
dense_3 (Dense)                  (None, 100)           50100       dropout_2[0][0]
____________________________________________________________________________________________________
dropout_3 (Dropout)              (None, 100)           0           dense_3[0][0]
____________________________________________________________________________________________________
dense_4 (Dense)                  (None, 50)            5050        dropout_3[0][0]
____________________________________________________________________________________________________
dropout_4 (Dropout)              (None, 50)            0           dense_4[0][0]
____________________________________________________________________________________________________
dense_5 (Dense)                  (None, 1)             51          dropout_4[0][0]
====================================================================================================
Total params: 4,062,321
Trainable params: 4,062,321
Non-trainable params: 0
____________________________________________________________________________________________________
Epoch 1/30
I tensorflow/core/common_runtime/gpu/gpu_device.cc:885] Found device 0 with properties:
name: Tesla P100-SXM2-16GB
major: 6 minor: 0 memoryClockRate (GHz) 0.405
pciBusID 0000:8a:00.0
Total memory: 15.90GiB
Free memory: 15.62GiB
I tensorflow/core/common_runtime/gpu/gpu_device.cc:906] DMA: 0
I tensorflow/core/common_runtime/gpu/gpu_device.cc:916] 0:   Y
I tensorflow/core/common_runtime/gpu/gpu_device.cc:975] Creating TensorFlow device (/gpu:0) -> (device: 0, name: Tesla P100-SXM2-16GB, pci bus id: 0000:8a:00.0)
Epoch 1/30
2113709/2113700 [==============================] - 17376s - loss: 0.1550 - val_loss: 0.1306
Epoch 2/30
2113748/2113700 [==============================] - 17110s - loss: 0.1204 - val_loss: 0.1110
Epoch 3/30
2113737/2113700 [==============================] - 17116s - loss: 0.1030 - val_loss: 0.0921
Epoch 4/30
2113712/2113700 [==============================] - 17087s - loss: 0.0890 - val_loss: 0.0797
Epoch 5/30
2113709/2113700 [==============================] - 17230s - loss: 0.0782 - val_loss: 0.0701
Epoch 6/30
2113736/2113700 [==============================] - 17198s - loss: 0.0700 - val_loss: 0.0639
Epoch 7/30
2113730/2113700 [==============================] - 17219s - loss: 0.0639 - val_loss: 0.0580
Epoch 8/30
2113728/2113700 [==============================] - 17156s - loss: 0.0592 - val_loss: 0.0539
Epoch 9/30
2113703/2113700 [==============================] - 17145s - loss: 0.0555 - val_loss: 0.0510
Epoch 10/30
2113738/2113700 [==============================] - 17103s - loss: 0.0528 - val_loss: 0.0515
Epoch 11/30
2113723/2113700 [==============================] - 17754s - loss: 0.0506 - val_loss: 0.0477
Epoch 12/30
2113711/2113700 [==============================] - 18338s - loss: 0.0489 - val_loss: 0.0472
Epoch 13/30
2113732/2113700 [==============================] - 18728s - loss: 0.0473 - val_loss: 0.0469
Epoch 14/30
2113745/2113700 [==============================] - 18740s - loss: 0.0458 - val_loss: 0.0445
Epoch 15/30
2113754/2113700 [==============================] - 18613s - loss: 0.0448 - val_loss: 0.0439
Epoch 16/30
2113711/2113700 [==============================] - 18552s - loss: 0.0437 - val_loss: 0.0434
Epoch 17/30
2113740/2113700 [==============================] - 18645s - loss: 0.0428 - val_loss: 0.0425
Epoch 18/30
2113743/2113700 [==============================] - 18649s - loss: 0.0420 - val_loss: 0.0420
Epoch 19/30
2113703/2113700 [==============================] - 18796s - loss: 0.0412 - val_loss: 0.0425
Epoch 20/30
2113748/2113700 [==============================] - 18403s - loss: 0.0406 - val_loss: 0.0423
Epoch 21/30
2113746/2113700 [==============================] - 18722s - loss: 0.0401 - val_loss: 0.0412
Epoch 22/30
2113752/2113700 [==============================] - 18599s - loss: 0.0395 - val_loss: 0.0419
Epoch 23/30
2113715/2113700 [==============================] - 18243s - loss: 0.0389 - val_loss: 0.0409
Epoch 24/30
2113710/2113700 [==============================] - 18734s - loss: 0.0384 - val_loss: 0.0419
Epoch 25/30
2113744/2113700 [==============================] - 18820s - loss: 0.0380 - val_loss: 0.0399
Epoch 26/30
2113719/2113700 [==============================] - 18118s - loss: 0.0375 - val_loss: 0.0400
Epoch 27/30
2113753/2113700 [==============================] - 18785s - loss: 0.0372 - val_loss: 0.0397
Epoch 28/30
2113706/2113700 [==============================] - 18752s - loss: 0.0368 - val_loss: 0.0398
Epoch 29/30
2113749/2113700 [==============================] - 18731s - loss: 0.0364 - val_loss: 0.0405
Epoch 30/30
2113709/2113700 [==============================] - 18683s - loss: 0.0361 - val_loss: 0.0388
```
