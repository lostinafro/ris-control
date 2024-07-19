# ris-control

Code implementing the performance evaluation of the paper "On the Impact of Control Signaling in RIS-Empowered Wireless Communications"

## Abstract
> The research on Reconfigurable Intelligent Surfaces (RISs) has dominantly been focused on physical-layer aspects and analyses of the achievable adaptation of the wireless propagation environment. Compared to that, questions related to system-level integration of RISs have received less attention. We address this research gap by analyzing the necessary control/signaling operations that are necessary to integrate RIS as a new type of wireless infrastructure element. We build a general model for evaluating the impact of control operations along two dimensions: i) the allocated bandwidth of the control channels (in-band and out-of-band), and ii) the rate selection for the data channel (multiplexing or diversity). Specifically, the second dimension results in two generic transmission schemes, one based on channel estimation and the subsequent optimization of the RIS, while the other is based on sweeping through predefined RIS phase configurations. We analyze the communication performance in multiple setups built along these two dimensions. While necessarily simplified, our analysis reveals the basic trade-offs in RIS-assisted communication and the associated control operations. The main contribution of the paper is a methodology for systematic evaluation of the control overhead in RIS-aided networks, regardless of the specific control schemes used. 

The paper is published in open access on the [IEEE Open Journal of the Communications Society](https://ieeexplore.ieee.org/document/10600711).

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
 
