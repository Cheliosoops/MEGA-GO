# MEGA-GO
# MEGA-GO: Functions prediction of diverse protein sequence length using MULTI-SCALE GRAPH ADAPTIVE NEURAL NETWORK
![image](https://github.com/Cheliosoops/MEGA-GO/blob/main/model.png)
## Usage
### Data preparation
Due to the large amount of the protein data, data is not provided here, it is recommended to refer to https://github.com/flatironinstitute/DeepFRI.
### Environment preparation
We have the environment.yml, to set up please run "**conda env create --file environment.yml**". <br />
Please check your python version and cuda version carefully, our python and cuda versions are **Python 3.10.12** and **11.7** respectively.
### Model training
Code are placed at the directory namely the MEGA-GO, you can simply run "**python trian.py**", once the data is ready. <br />
'--task', (choose bp mf cc the GO task);<br />
'--suffix', (the save model name);<br />
'--device', (the device index);<br />
'--pooling', (the operation mode in adaSAB);<br />
'--batch_size'.<br />

