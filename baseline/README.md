## Configuration

* Modify `data_dir` in `experiments.conf` to your local path that will contain all datasets, models, logs, results.

* Adjust number of epochs, learning rate, etc. in `experiments.conf`

## Run training

`python run_qa.py [config] [gpu]`

To train mT5 on TyDiQA on GPU 0, it is

`python run_qa.py mt5_large_zero_shot_tydiqa 0`

Make sure the `data_dir` property in `experiments.conf` has the corresponding path on your machine.

## Evaluate

`python evaluate.py [config] [model_id] [gpu] [dataset]`

To evaluate mT5 on TyDiQA on GPU 0 with trained model ID `Jan25_01-27-23`, it is

`python evaluate.py mt5_large_zero_shot_tydiqa Jan25_01-27-23 0 tydiqa`
