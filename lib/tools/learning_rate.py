import numpy_ml as nml;
import numpy as np;
def exp_lr_func(initial_lr: float, stage_length = 1, staircase: bool = False, decay: float = 0.99):
    return nml.neural_nets.schedulers.ExponentialScheduler(initial_lr = initial_lr, 
                stage_length = stage_length, staircase = staircase, decay = decay).learning_rate

def linear_lr_func(total_timesteps: float, step_infos):
        def step(step)->float:
            tmp_rate = 0;
            for initial_lr, min_lr, rate in step_infos:
                tmp_tol = rate * total_timesteps;
                step = step - tmp_rate * total_timesteps;
                if step <= tmp_tol:
                    return np.maximum((1.0 - step / tmp_tol) * (initial_lr - min_lr) + min_lr, np.zeros_like(step))
                tmp_rate += rate;
            return np.maximum(min_lr, np.zeros_like(step))
        return step  