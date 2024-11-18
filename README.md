# 16831_FMRL_project
implementation based on https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10339821

See TODOs in run_frml.py

### Setting up conda env:

NOTE: If you find that you need to add additional dependencies via snap or pip or whatever your OS
uses, please remember to add them to environemnt.yaml so we all share the same updates

```
conda activate rob831
```

```
conda env update --file environment.yaml
```

### Running main script:

See add_args() in run_frml.py for additional options. Note that the original paper uses 200 envs,
I'd like to avoid vectorizing envs initially as I've found this can come with it's own challenges 
when accessing shared memory

```
python run_frml.py --env_name HalfCheetah --env_goal random_direction --n_client_envs 20
```
