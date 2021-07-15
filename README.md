# iFedAvg
iFedAvg -  Interpretable Data-Interoperability for Federated Learning

The paper can be viewed on Arxiv [at this link](https://arxiv.org/abs/2107.06580).

### Executing the code

First, ensure that all dependencies in `requirements.txt` are satisfied. 

To run the code, execute `python main.py` with the following command line arguments:
- `--config_path`: path to the configuration file, such as `configs/sample.yml`
- `--gpu`: (optional) flag to enable GPU computation (using CUDA, when available)

### Generating outputs 
It is possible to track the training using tensorboard. To launch it, run the following command:
```python
tensorboard --logdir outputs/<dataset_name>/<experiment_name>
```
The `dataset_name` and `experiment_name` are defined in the config file with which you launched the experiment.

To generate the outputs visible in the paper, two notebooks are provided:
- `notebooks/post-processing.ipynb`: can be directed to the **experiment directory** and will compare different runs or methods (such as iFedAvg and APFL).
- `notebooks/single_run_analysis.ipynb`: can be directed to a **run directory** and will generate the interpretable outputs for iFedAvg.

### Citations
If you wish to cite this repository or paper, please use the bibtext below:

```
@article{roschewitz2021ifedavg,
  title={iFedAvg: Interpretable Data-Interoperability for Federated Learning},
  author={David Roschewitz and Mary-Anne Hartley and Luca Corinzia and Martin Jaggi},
  journal={arXiv preprint arXiv:2107.06580},
  year={2021},
  primaryClass={cs.LG}
}
```

---

### Additional functionality

This code base has additional functionality, which is not outlined in the paper and was used for empirial testing. 
In the future, documentation on how to use this part might be added.