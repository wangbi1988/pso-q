import numpy as np

class cpuPSOMP():
    def __init__(self, C1, C2, W, N, D):
        self.D = D;
        self.N = N;
        self.C1, self.C2, self.W = C1, C2, W;
        self.boundary = None;
        self.one_hot_eye = None;
                                        
    def minimize(self, y, bounds, batch_size, maxiter, func):
        expanded_y = self.tile_reshape(y, tiles = [1, self.N], shape = (batch_size * self.N, np.shape(y)[-1]));
        if self.boundary is None:
            self.boundary = np.stack([bounds[0], bounds[1]]);
        boundary = self.boundary.copy();

        infos = (np.random.rand(batch_size * self.N, self.D,), 
                 np.zeros(shape = (batch_size * self.N, self.D)), 
                 np.zeros((batch_size, self.D,)), 
                 np.ones((batch_size, 1)) * -1e+8);

        for i in range(maxiter):
            infos = self.loop_(infos, expanded_y, func, batch_size, boundary);
        p0, _, p1, p2 = infos;
        
        p1 = np.reshape(p1, (batch_size, 1, self.D,));
        p2 = np.reshape(p2, (batch_size, 1, ));
        return p1, p2

    
    def tile_reshape(self, x, tiles, shape):
        processed_y = np.tile(x, tiles);
        expanded_y = np.reshape(processed_y, shape);
        return expanded_y;

    def one_hot(self, argmins):
        if self.one_hot_eye is None:
            self.one_hot_eye = np.eye(self.N);
        tmp = self.one_hot_eye[argmins];
        return tmp
            
    def inner_loop(self, particles, expanded_y, func, batch_size):
        evs = func(particles, expanded_y);
        reshaped_evs = np.reshape(evs, (batch_size, self.N));
        reshaped_particles = np.reshape(particles, (batch_size, self.N, self.D));
        #0.4~0.5
        argmins = np.argmax(reshaped_evs, axis = -1);
        idx_ = self.one_hot(argmins)
        local_eval = np.sum(reshaped_evs * idx_, axis = -1, keepdims = True);
        local_particle = np.sum(reshaped_particles * np.reshape(idx_, (batch_size, self.N, 1)), axis = 1);
        return local_eval, local_particle;
            

            
    def loop_(self, infos, expanded_y, func, batch_size, boundary):
        particles, velocities, global_particle, global_eval = infos;
        local_eval, local_particle = self.inner_loop(particles, expanded_y, func, batch_size);
        
        cond1 = np.greater(global_eval, local_eval);

        global_particle = np.where(cond1, global_particle, local_particle);
        global_eval = np.where(cond1, global_eval, local_eval);
        # global_particle = global_particle * cond1 + local_particle * (1 - cond1);
        # global_eval = global_eval * cond1 + local_eval * (1 - cond1);
        
        tmp_local_particle =  self.tile_reshape(local_particle, 
                                tiles = [1, self.N], shape = (batch_size * self.N, self.D));
        tmp_global_particle =  self.tile_reshape(global_particle, 
                                tiles = [1, self.N], shape = (batch_size * self.N, self.D));
        
        local_diff = tmp_local_particle - particles;
        global_diff = tmp_global_particle - particles;
        
        r1r2 = np.random.uniform(size = (2, batch_size * self.N, self.D));
        tp1 = self.W * velocities;
        tp2 = self.C1 * r1r2[0] * local_diff;
        tp3 = self.C2 * r1r2[1] * global_diff;
        velocities = tp1 + tp2 + tp3;
        
        particles += velocities;
        particles = np.where(np.logical_or(np.less(particles, boundary[0]), 
                                            np.greater(particles, boundary[1])), 
                                            np.random.uniform(size = (batch_size * self.N, self.D,)), particles);

        return particles, velocities, global_particle, global_eval;