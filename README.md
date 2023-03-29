# ris-control

Code implementing the performance evaluation of the paper "A Framework for Control Channels Applied to Reconfigurable Intelligent Surfaces"

## Abstract
> The research on Reconfigurable Intelligent Surfaces (RISs) has dominantly been focused on physical-layer aspects and analyses of the achievable adaptation of the propagation environment. Compared to that, the questions related to link/MAC protocol and system-level integration of RISs have received much less attention. This paper addresses the problem of designing and analyzing control/signaling procedures, which are necessary for the integration of RISs as a new type of network element within the overall wireless infrastructure. We build a general model for designing control channels along two dimensions: i) allocated bandwidth (in-band and out-of band) and ii) rate selection (multiplexing or diversity). Specifically, the second dimension results in two transmission schemes, one based on channel estimation and the subsequent adapted RIS configuration, while the other is based on sweeping through predefined RIS phase profiles. The paper analyzes the performance of the control channel in multiple communication setups, obtained as combinations of the aforementioned dimensions. While necessarily simplified, our analysis reveals the basic trade-offs in designing control channels and the associated communication algorithms. Perhaps the main value of this work is to serve as a framework for subsequent design and analysis of various system-level aspects related to the RIS technology.

The paper is submitted to Transactions on Wireless Communications.
A preprint version can be found [here](http://arxiv.org/)

The main goodput performance are obtainable by running (in this order)
```
opt_ce_vs_tau.py
cb_bsw_vs_minimum_snr.py
cb_bsw_vs_tau.py
```
Note that the scripts should be run with the flag ```-r``` to save the ``.npz`` results into the ``data`` folder.

Visualization of the already presented results on the goodput can be obtained running ```plot_goodput_vs_tau.py```.
Also in this case flag ```-r``` print the results, in both .jpg and .tex format, in a ```plots``` directory.

Based on the previous results, the impact of control channel results are obtainable by running
```
plot_utility_vs_pcc.py
cc_reliability.py
```

The standard parameters can be edited changing ```environment.py```. 
Check and edit ```scenario.common.standard_output_dir``` to change the default saving folder.
 

