# DeviceAware-Ternary-Crossbar

This project is for Algorithm-Hardware Co design project.


we have some files which is .py and for running and I'll describe here what is inside each file


----
### 01_First_ANN.py

first we load train and test MNIST dataset and we have 3 layer (784-256-128) ANN and we trained 100 epoch with 16 batch size and we recieved 98 % accuracy and we save that in **Pth_Models/**  wiuth name mnist_ann_{date}.pth file which is our digital ANN model.


so we have mnist_ann_07_sep_2026.pth that is Base ANN


----
### 02_TNN_Optimization.py

we load again Train and test dataset of MNIST and then we create TernaryLinear layer and TernaryMNIST model 


we save TNN without BP and also  try (not save) different back propagation strategy and plot different strategies here.


it doesnt save any TNN models because we just want to find the best strategies



----
### 02b_BNN_optimization.py

Here we trained Binary neural network and we saved with 07_sep_2026_BNN.pth


----
### 03_Base_ANN_TNN.py

In previous files we tried to find the best ANN and TNN and we reach teh best configuration. so here we again train ANN , TNN .


ANN is 3 linear (784-256-128) relu and is float/full precision weight and is our normal baseline.

TNN is used TernaryLinear we have two weight one is trainable/latent/hidden which is layer weeight it is float and also we have ternary in forward which go from -1,0,+1



so we have these 3 .pth files

- BASE_ANN_mnist_lr_epoch_seed_tme.pth
- BASE_TNN_mnist_lr_epoch_seed_tme.pth
- TERNARY_ONLY_mnist_lr_epoch_seed_tme.pth


in 07 sep we have

- BASE_ANN_mnist_lr0.0001_ep100_seed42_20260908.pth
- BASE_TNN_mnist_smooth_tw_w0.7_lr0.0001_ep100_seed42_20260908.pth
- TERNARY_ONLY_mnist_tw0.7_th0.05_seed42_20260908.pth

----
### 04_Load_ANN_TNN.py

In previous file we trained base optimal ANN, TNN , Inference TNN and then in this file we load again these models and we got some plots to bwe sure .


** : change ANN_CHECKPOINT and TNN_CHECKPOINT based on name that we saved 


----
### 05_device_modeling.py


section1 : Programming error mdoeling 
First in device_data and drift data that we have for device 3 , we have the current
of On and   Off state for 5 repttion(at t<1 stable) so we can get mean of that and this is our 
TARGET ON current and TARGET OFF Current . and because we have our read voltage 
we can calculate R_on and R_of


Then we go for I-V data from APM_Datafinal.xlsx that we have 100 columns (50 sweeps) 
so from each sweep we see on, ofr current at Vread= +1 V
so for each sweeep with np.gradient(V) we seperate forward and backward 
then we get interpolation at 1v . the bigger is On, the smaller is OFF.
then we can get programming error 

On_error = Ion_actual - TARGET_ON
OFF_Error = Ioff_actual - TARGET_OFF
 


R min , Rmax , Rmin comes from here
and also  we got sigma relative means that when you switch on on or off
you didn't get exact number, you get with some standard deviation



It help us to create Cross-sim package

----
### 06_Ideal_Cross_sim.py


Here with CrossSim we just create digital ANN (bitsliced) and Analog TNN (balanced)
only to see that everything is perfect and because it is ideal we must not see any
things difference to confirm that Cross Sim is sync with torch

we access cross sim with

```python
CROSS_SIM_DIR = Path("/Users/apm/Desktop/tern-net/cross-sim")
```


change ANN_CHECKPOINT and TNN_CHECKPOINT


we load Ideal ANN and TNN . we load MNIST Data set.

we also create bitsliced ANN and also balanced TNN.

so we have 4 things.

we got accuracy and some plots to compare these things together.



----
### 07_Real_Cross_Sim_TNN_vs_ANN.py

after modeling programmoing error and drift in file 05 now we go ideal ANN and TNN to see the digital and analog si the same (in ideal) now we consider this non ideality.


Now in 07 we want to consider Rmin, Rmax and also Programming error
to consider that our programming error is ok or not.


First we have ideal (from 06) for ANN and TNN.
We implement SONOS and APM to compare with each others. we found APM is reasonable

then we compare the TNN and ANN in terms of configurations. and we decide
to go with TNN.
    



----
### 08_Real_TNN_Cross_S.py
we load MNISR , we load digital ANN and TNN 

and then we create TNN cross sim (real (non ideal)) with apm model

analzye programming error and drifts.




----
### 09_Final_TNN_Dev_ADC_DAC_Paraisitic.py

Until now we only consider poragmming error and also drift error, but we can now consider also ADC/DAC errors and we optimize between different configuration for adc and dac.







----
### 10_Device_Aware_Training.py

Here we have loaders and also we have ternaryWeightfunctiion class and Ternalylinear class and TernaryMNIST mdoel which is our final model.


we have digital TN training and we saved with **date_digital_tnn.pt**


and then we have real device (with programming, drfit error and also adc/dac and all of them) and we devcie-aware train 
and saved with **date_device_aware_tnn.pt**






