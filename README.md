# Temporal Logistic Neural BoF method for Limit Order Book data analysis
Implementation of the Temporal Logistic Neural BoF method

This repository demonstrates how to use the Temporal Logistic Neural BoF method to classify limit order book data. Please  download the preprocessed data from [here](https://www.dropbox.com/s/vvvqwfejyertr4q/lob.tar.xz?dl=0) and place the .h5 file into the data folder. You can then run the experiments by executing the [run_exps.py](https://github.com/passalis/finbof/blob/master/run_exps.py) file. Then you can print the results by running the [print_results.py](https://github.com/passalis/finbof/blob/master/print_results.py) script. The proposed model is implemented in [bof_models.py](https://github.com/passalis/finbof/blob/master/models/bof_models.py), while training utilities are provided in [lob_utils](https://github.com/passalis/finbof/tree/master/lob_utils).

If you use this code in your work please cite the following paper:

<pre>
@article{temporal-bof,
        title       = "Temporal Logistic Neural Bag-of-Features for Financial Time series Forecasting leveraging Limit Order Book Data",
	author      = "Passalis, Nikolaos and Tefas, Anastasios and Kanniainen, Juho and Gabbouj, Moncef and Iosifidis, Alexandros",
	journal   = "Pre-print submitted to Pattern Recognition Letters",
	
}
</pre>
