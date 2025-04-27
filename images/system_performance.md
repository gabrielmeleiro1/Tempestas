First talk about how the calculation of the approximate energy proxy was useful (because energy production increase by a lot in the netherlands)


"images/calculated_energy_output.png"





Energy output prediction table


Metric          | TCN Model       | Persistence (1H)  | Persistence (24H)  | Simple Average 
----------------|-----------------|-------------------|--------------------|-----------------
MAE (MW)        | 110.86          | 112.54            | 842.01             | 836.85         
RMSE (MW)       | 164.82          | 177.38            | 1186.74            | 1137.85        








"images/model_loss.png" - This is the graph for both Scaled normalized MSE and MAE;


"images/actual_vs_predicted_MW.png" - You will interpret









Energy price prediction table:

Persistence Baseline (predict P(t)=P(t-1)):
  MAE:  15.179 EUR/MWh
  RMSE: 21.529 EUR/MWh

Average Baseline (predict P(t)=mean(P_train)):
  MAE:  36.865 EUR/MWh
  RMSE: 47.482 EUR/MWh
XGBoost	Persistence	Average
Metric			
MAE	7.942	15.179	36.865
RMSE	12.223	21.529	47.482


"images/energy_price_prediction.png"

