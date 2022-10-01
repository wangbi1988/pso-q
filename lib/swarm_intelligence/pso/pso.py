# -*- coding: utf-8 -*-
"""
Created on Mon Nov  9 08:54:07 2020

@author: bb
"""


import numpy as np;
# import tensorflow as tf;
import tensorflow as tf
if not tf.__version__.startswith('1'):
    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()

class PSOMP:
    def __init__(self, C1, C2, W):
        self.C1, self.C2, self.W = C1, C2, W;
    
    def minimize(self, N, D, maxiter, boundary,
                 func, para_key = None, paras_dict = {}):
        global_particle, global_eval = None, np.Inf;
        local_particle, local_eval = None, np.Inf;
        
        particles = np.random.randn(N, D);
        velocities = np.zeros_like(particles);
        
        boundary = np.array(boundary);
        
        norm_factors = boundary[1] - boundary[0];
        
        for i in range(maxiter):
            if para_key is None:
                evals = func(particles);
            else:
                paras_dict[para_key] = particles;
                evals = func(paras_dict);
                
            argmin = np.argmin(evals);
            local_particle, local_eval = particles[argmin].copy(), evals[argmin];
            if local_eval <= global_eval:
                global_particle, global_eval = local_particle.copy(), local_eval;
            
            r1, r2 = np.random.random(2);
            
            local_diff = local_particle - particles;
            global_diff = global_particle - particles;
            
            velocities = self.W * velocities + \
                        self.C1 * r1 * local_diff + \
                        self.C2 * r2 * global_diff;
            
            particles += velocities;
            
            
            particles = np.clip((particles - boundary[0]) / norm_factors, 0, 1);
            particles = particles * norm_factors + boundary[0];
            
        return global_particle, global_eval;
    
class tfPSOMP(PSOMP):
    def __init__(self, C1, C2, W, N, D):
        with tf.variable_scope("swarm_intelligence"):
            self.D = D;
            self.N = N;
            self.C1, self.C2, self.W = (tf.constant(C1, dtype = tf.float32, name = 'PSO.C1'), 
                                        tf.constant(C2, dtype = tf.float32, name = 'PSO.C2'), 
                                        tf.constant(W, dtype = tf.float32, name = 'PSO.W'));
                                        
    def minimize(self, ops, sess, paras_dict = {}):
        loop = sess.run([ops], feed_dict = paras_dict);
        return loop
    
    @tf.function
    def build_batch_v4(self, y, maxiter, func, bounds, batch_size, k = 1):
        with tf.variable_scope("swarm_intelligence"):
            @tf.function
            def tile_reshape(x, tiles, shape):
                processed_y = tf.tile(x, tiles);
                expanded_y = tf.reshape(processed_y, shape = shape);
                return expanded_y;
            
            @tf.function
            def inner_(particles, expanded_y, func, batch_size, N, D):
                evs = func(particles, expanded_y);
                reshaped_evs = tf.reshape(evs, shape = (batch_size, self.N));
                reshaped_particles = tf.reshape(particles, shape = (batch_size, self.N, self.D));
                #0.4~0.5
                argmins = tf.argmax(reshaped_evs, axis = -1, output_type = tf.int32);
                print(argmins)
                idx_ = tf.one_hot(argmins, N)
                local_eval = tf.reduce_sum(reshaped_evs * idx_, axis = -1);
                local_particle = tf.reduce_sum(reshaped_particles * tf.reshape(idx_, shape = (batch_size, self.N, 1)), axis = 1);
                return local_eval, local_particle;
            
            
            @tf.function
            def loop_(i, maxiter, infos):
                particles, velocities, global_particle, global_eval = infos;
                local_eval, local_particle = inner_(particles, expanded_y, 
                                                            func, batch_size, self.N, self.D);
                print(local_eval, local_particle, global_eval, global_particle)
                #tf.print(local_eval)
                
                global_particle = tf.where(tf.greater(global_eval, local_eval), 
                                           global_particle, local_particle);
                global_eval = tf.where(tf.greater(global_eval, local_eval), 
                                       global_eval, local_eval);
                
                tmp_local_particle =  tile_reshape(local_particle, tiles = [1, self.N], 
                                                   shape = (batch_size * self.N, self.D));
                tmp_global_particle =  tile_reshape(global_particle, tiles = [1, self.N], 
                                                   shape = (batch_size * self.N, self.D));
                
                local_diff = tmp_local_particle - particles;
                global_diff = tmp_global_particle - particles;
                
                r1r2 = tf.random.uniform((2, batch_size * self.N, self.D), minval = 0, maxval = 1);
                tp1 = self.W * velocities;
                tp2 = self.C1 * r1r2[0] * local_diff;
                tp3 = self.C2 * r1r2[1] * global_diff;
                velocities = tp1 + tp2 + tp3;
                
                #particles = tf.clip_by_value(particles + velocities, boundary[0], boundary[1]);
                particles += velocities;
                particles = tf.where(tf.greater(particles, boundary[1]), init_pos, particles);
                particles = tf.where(tf.less(particles, boundary[0]), init_pos, particles);
                return i + 1, maxiter, (tf.stop_gradient(particles), 
                                        tf.stop_gradient(velocities), 
                                        tf.stop_gradient(global_particle), 
                                        tf.stop_gradient(global_eval));
        
            
            expanded_y = tile_reshape(y, tiles = [1, self.N], shape = (batch_size * self.N, tf.shape(y)[-1]));
            print('expanded_y', expanded_y)
            
            boundary = tf.stack([bounds[0], bounds[1]], name = 'PSO.boundary'); 
            init_pos = tf.random.uniform((batch_size * self.N, self.D,), 
                                         minval= boundary[0], maxval= boundary[1], 
                                         name = 'PSO.init_particles', dtype = tf.float32);
            print('init_pos', init_pos)
            
            
        _, _, (p0, _, p1, p2) = tf.while_loop(lambda a, b, _: tf.less(a, b), 
               loop_, [tf.constant(0), 
                       tf.constant(maxiter, dtype = tf.int32, name = 'PSO.maxiter'), 
                       (tf.constant(np.random.rand(batch_size * self.N, self.D,), dtype = tf.float32, name = 'PSO.particles'), 
                        tf.constant(np.zeros(shape = (batch_size * self.N, self.D)), 
                                    dtype = tf.float32, name = 'PSO.velocities'), 
                        tf.constant(np.zeros((batch_size, self.D,)), dtype = tf.float32, 
                                    name = 'PSO.global_particle'), 
                        tf.constant(np.ones((batch_size)) * -1e+8, dtype = tf.float32, 
                                    name = 'PSO.global_eval'))]);
        
        if k > 1:
            slice_ = slice(int(self.N / k) - 1, self.N, int(self.N / k));
            print(slice_)
            evs = func(p0, expanded_y);
            reshaped_evs = tf.reshape(evs, shape = (batch_size, self.N, 1, 1));
            reshaped_particles = tf.reshape(p0, shape = (batch_size, self.N, self.D, 1));
            arg_sort = tf.argsort(reshaped_evs, axis = 1, 
                                  #direction = 'DESCENDING', stable = True, 
                                  name = 'PSO.argsort');
            index_for_random_k = arg_sort[:, slice_];
            
            idx_ = tf.one_hot(index_for_random_k, self.N);
            new_idx_ = tf.reshape(idx_, shape = (batch_size, k, self.N, 1));
            transposed_idx_ = tf.transpose(new_idx_, perm = (0, 2, 3, 1));
            print(new_idx_)
            
            k_particles = reshaped_particles * transposed_idx_;
            print(k_particles)
            sumed_k_particles = tf.transpose(tf.reduce_sum(k_particles, axis = 1), perm = (0, 2, 1));
            print(sumed_k_particles)
            
            k_eval = reshaped_evs * transposed_idx_;
            print(k_eval)
            sumed_k_eval = tf.reduce_sum(tf.reduce_sum(k_eval, axis = 1), axis = 1);
            print(sumed_k_eval)
            return tf.stop_gradient(sumed_k_particles), tf.stop_gradient(sumed_k_eval)
        else:
            return tf.stop_gradient(p1), tf.stop_gradient(p2)