# VAE-DCGAN

### Install with virtualenv
```bash=
python3 -m virtualenv venv
source venv/bin/activate
pip3 install -r requirements.txt
```

### Run
```bash=
python3 main.py --dataset celeba --dataroot [path to dataset] --cuda
python3 interpolate.py --dataroot [path to dataset] --netE [path to NetE] --netG [path to NetG] --cuda
```

### Result
#### VAE vs VAE/DEGAN
![image](https://github.com/ballinhuang/VAE-DCGAN/blob/master/Final/result/vae_exp.png)
![image](https://github.com/ballinhuang/VAE-DCGAN/blob/master/Final/result/vae_dcgan_exp.png)

#### Interpolate Visual Attribute Vectors with AE-DCGAN
![image](https://github.com/ballinhuang/VAE-DCGAN/blob/master/result/exp_result.png)
