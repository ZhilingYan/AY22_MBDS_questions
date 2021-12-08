source activate imageClassifer


model_path=model_weights__2021_12_07__18_13_26__1.pth
python test.py --init_model_file saved_models/$model_path
