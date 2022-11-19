# -*- coding: utf-8 -*-
"""
Created on Mon Nov  9 08:54:07 2020

@author: bb
"""


#import tf_util;
from . import tf_util;
import numpy as np;
import tensorflow as tf;
from tf_slice_assign import slice_assign;
    
class tfPSOMP():
    def __init__(self, C1, C2, W, N, D):
        with tf.variable_scope("swarm_intelligence"):
            self.D = D;
            self.N = N;
            self.C1, self.C2, self.W = (tf.constant(C1, dtype = tf.float32, name = 'PSO.C1'), 
                                        tf.constant(C2, dtype = tf.float32, name = 'PSO.C2'), 
                                        tf.constant(W, dtype = tf.float32, name = 'PSO.W'));
                                        
            self.global_particle, self.global_eval = tf.Variable(tf.zeros((D,)), dtype = tf.float32, name = 'PSO.global_particle', trainable = False), \
                                                        tf.Variable(np.inf, dtype = tf.float32, name = 'PSO.global_eval', trainable = False);
            self.local_particle, self.local_eval = tf.Variable(tf.zeros((D,)), dtype = tf.float32, name = 'PSO.local_particle', trainable = False), \
                                                        tf.Variable(np.inf, dtype = tf.float32, name = 'PSO.local_eval', trainable = False);
            
            self.particles = tf.Variable(tf.random.uniform((N, D), minval= 0, maxval= 1), 
                                    dtype = tf.float32, name = 'PSO.particles', trainable = False);
            self.velocities = tf.Variable(tf.zeros(shape = (N, D)), 
                                    dtype = tf.float32, name = 'PSO.velocities', trainable = False);
        
    def build_single(self, maxiter, func, bounds, batch_size = 1):
        with tf.variable_scope("swarm_intelligence"):
            i = tf.Variable(0, dtype = tf.int32, name = 'PSO.counter');
            reset = tf.Variable(True, dtype = tf.bool, name = 'PSO.reset_flag');
            maxiter = tf.constant(maxiter, dtype=tf.int32, name = 'PSO.maxiter');
            
            boundary = np.array(bounds);
            norm_factors = tf.constant(boundary[1] - boundary[0], dtype = tf.float32, name = 'PSO.norm_factors');
            boundary = tf.constant(boundary, dtype = tf.float32, name = 'PSO.boundary');
            
        def body(i, maxiter, reset, infos):
            global_particle, global_eval, local_particle, local_eval, particles, velocities = infos;
            
            global_eval, local_eval, particles, velocities = \
                tf.cond(reset, 
                        false_fn = lambda: (global_eval, local_eval, particles, velocities), 
                        true_fn = lambda: (np.inf, np.inf, tf.random.uniform(particles.shape, minval= 0, maxval= 1) * \
                                                                norm_factors + boundary[0], tf.zeros(shape = velocities.shape)));
            
            reset = tf.logical_and(reset, tf.logical_not(reset));
            local_diff = local_particle - particles;
            global_diff = global_particle - particles;
            
            evals = func(particles);
            
            argmin = tf.argmin(evals, -1);
            
            tmp_local_particle, tmp_local_eval = particles[argmin], evals[argmin];
            
            local_particle, local_eval = tmp_local_particle, tmp_local_eval;
            global_particle, global_eval = tf.cond(tf.greater(global_eval, local_eval),
                                                        true_fn = lambda: [tmp_local_particle, tmp_local_eval], 
                                                        false_fn = lambda: [global_particle, global_eval]);
                                                   
            r1r2 = tf.random.uniform((2,), minval= 0, maxval= 1);
            velocities = self.W * velocities + \
                                self.C1 * r1r2[0] * local_diff + \
                                self.C2 * r1r2[1] * global_diff;
            
            norm_particles = (particles + velocities - boundary[0]) / norm_factors;
            
            clipped_norm_particles = tf.clip_by_value(norm_particles, 0, 1);
            reversed_norm_particles = clipped_norm_particles * norm_factors + boundary[0];
            
            particles = reversed_norm_particles;
            
            return i + 1, maxiter, reset, (global_particle, global_eval, local_particle, local_eval, particles, velocities);
        
        
        self.loop = tf.while_loop(lambda a, b, c, d: tf.less(a, b), body, [i, maxiter, reset,
                                  (self.global_particle, self.global_eval, 
                                   self.local_particle, self.local_eval, 
                                   self.particles, self.velocities)]);
        return self.loop;
    
    def build_batch(self, y, maxiter, func, bounds, batch_size):
        with tf.variable_scope("swarm_intelligence"):
            i = tf.Variable(0, dtype = tf.int32, name = 'PSO.iter_counter');
            j = tf.Variable(0, dtype = tf.int32, name = 'PSO.batch_counter');
            reset = tf.Variable(True, dtype = tf.bool, name = 'PSO.reset_flag');
            maxiter = tf.constant(maxiter, dtype=tf.int32, name = 'PSO.maxiter');
            batch_size = tf.constant(batch_size, dtype=tf.int32, name = 'PSO.batch_size');
            records = tf.Variable(tf.zeros(shape = (batch_size, self.D)), dtype = tf.float32, name = 'PSO.records');
            
            boundary = np.array(bounds);
            norm_factors = tf.constant(boundary[1] - boundary[0], dtype = tf.float32, name = 'PSO.norm_factors');
            boundary = tf.constant(boundary, dtype = tf.float32, name = 'PSO.boundary');
            
        @tf.function
        def cond(i, limit, *args):
            return tf.less(i, limit);
        
        @tf.function
        def body_single(i, maxiter, reset, yi, infos):
            global_particle, global_eval, local_particle, local_eval, particles, velocities = infos;
            
            global_eval, local_eval, particles, velocities = \
                tf.cond(reset, 
                        false_fn = lambda: (global_eval, local_eval, particles, velocities), 
                        true_fn = lambda: (np.inf, np.inf, tf.random.uniform(particles.shape, minval= 0, maxval= 1) * \
                                                                norm_factors + boundary[0], tf.zeros(shape = velocities.shape)));
            
            reset = tf.logical_and(reset, tf.logical_not(reset));
            local_diff = local_particle - particles;
            global_diff = global_particle - particles;
            
            evals = func(particles, tf.repeat(tf.expand_dims(yi, axis = 0), self.N, axis = 0));
            
            argmin = tf.argmin(evals); 
            
            tmp_local_particle, tmp_local_eval = particles[argmin], evals[argmin];
            
            local_particle, local_eval = tmp_local_particle, tmp_local_eval;
            print(local_particle, local_eval)
            print(global_particle, global_eval)
            global_particle, global_eval = tf.cond(tf.greater(global_eval, local_eval),
                                                        true_fn = lambda: [tmp_local_particle, tmp_local_eval], 
                                                        false_fn = lambda: [global_particle, global_eval]);
            global_eval = tf.reshape(global_eval, ())
            r1r2 = tf.random.uniform((2,), minval= 0, maxval= 1);
            velocities = self.W * velocities + \
                                self.C1 * r1r2[0] * local_diff + \
                                self.C2 * r1r2[1] * global_diff;
            
            norm_particles = (particles + velocities - boundary[0]) / norm_factors;
            
            clipped_norm_particles = tf.clip_by_value(norm_particles, 0, 1);
            reversed_norm_particles = clipped_norm_particles * norm_factors + boundary[0];
            
            particles = reversed_norm_particles;
            
            return i + 1, maxiter, reset, yi, (global_particle, global_eval, 
                                           local_particle, local_eval, particles, velocities);
                                               
        @tf.function
        def body_batch(j, batch_size, records):
            print(self.local_particle, self.local_eval)
            print(self.global_particle, self.global_eval)
            print('+'*40)
            loop = tf.while_loop(cond, body_single, [i, maxiter, reset, 
                                 tf.gather(y, j),
                                  (self.global_particle, self.global_eval, 
                                   self.local_particle, self.local_eval, 
                                   self.particles, self.velocities)]);
            records = tf.tensor_scatter_nd_update(records, 
                                                  tf.expand_dims(tf.expand_dims(j, axis = 0), axis = 0), 
                                                  tf.expand_dims(loop[4][0], axis = 0))
            return j + 1, batch_size, records
        
        self.loop = tf.while_loop(cond, body_batch, [j, batch_size, records]);
        return self.loop;
                            
    def build_batch_v2(self, y, maxiter, func, bounds, batch_size):
        with tf.variable_scope("swarm_intelligence"):
            maxiter = tf.constant(maxiter, dtype = tf.int32, name = 'PSO.maxiter');
            
            global_particle = tf.Variable(tf.zeros((batch_size, self.D,)), dtype = tf.float32, name = 'PSO.global_particle', trainable = False);
            local_particle = tf.Variable(tf.zeros((batch_size, self.D,)), dtype = tf.float32, name = 'PSO.local_particle', trainable = False);
            
            global_eval = tf.Variable(tf.ones((batch_size, ), dtype = tf.float32) * np.inf, dtype = tf.float32, name = 'PSO.global_eval', trainable = False);
            local_eval = tf.Variable(tf.ones((batch_size, ), dtype = tf.float32) * np.inf, dtype = tf.float32, name = 'PSO.local_eval', trainable = False);
            
            boundary = np.array(bounds, dtype = np.float32);
            boundary = tf.stack([tf.repeat(tf.expand_dims(boundary[0], axis = 0), batch_size, axis = 0), 
                                             tf.repeat(tf.expand_dims(boundary[1], axis = 0), batch_size, axis = 0)], name = 'PSO.boundary'); 
            norm_factors = boundary[1] - boundary[0];
            
            particles = tf.Variable(tf.random.uniform((batch_size, self.D, self.N, ), minval= 0, maxval= 1), 
                                    dtype = tf.float32, name = 'PSO.particles', trainable = False) * tf.expand_dims(norm_factors, axis = 2) + tf.expand_dims(boundary[0], axis = 2);
            velocities = tf.Variable(tf.zeros(shape = (batch_size, self.D, self.N, )), 
                                    dtype = tf.float32, name = 'PSO.velocities', trainable = False);
            
                   
            @tf.function                 
            def subloop():
                evals_array = tf.TensorArray(dtype = tf.float32, size = batch_size);
                particles_array = tf.TensorArray(dtype = tf.float32, size = batch_size);
                #evals_array = tf.Variable([], dtype = tf.float32);
                #particles_array = tf.Variable([], dtype = tf.float32);
                for j in tf.range(batch_size):
                    tmp_p = tf.gather(particles, j);
                    ev = func(tf.transpose(tmp_p), 
                               tf.repeat(tf.expand_dims(tf.gather(y, j), axis = 0), 
                                         self.N, axis = 0));
                    ev = tf.cast(ev, tf.float32);
                    argmin = tf.argmin(ev, output_type = tf.int32);
                    print(argmin)
                    #tf.print(ev)
                    #0.25
                    #evals = tf.tensor_scatter_nd_update(evals, [[j]], tf.gather(evals, [j]));
                    #local_particle = tf.tensor_scatter_nd_update(local_particle, [j], tf.gather(tf.transpose(tmp_p, [1, 0]), j));
                    #0.3-0.1 = 0.2
                    #local_eval = tf.tensor_scatter_nd_update(local_eval, [[j]], tf.gather(ev, [[j]]));
                    evals_array = evals_array.write(j, ev[argmin[0]]);
                    particles_array = particles_array.write(j, tf.gather(tf.transpose(tmp_p, [1, 0]), argmin));
                    #evals_array = tf.concat([evals_array, tf.gather(ev, argmin)], axis = 0);
                    #particles_array = tf.concat([particles_array, tf.gather(tf.transpose(tmp_p, [1, 0]), argmin)], axis = 0);
                    #evals_array = evals_array.unstack(o.stack());
                    #particles_array = particles_array.unstack(m.stack());
                return tf.reshape(particles_array.stack(), (batch_size, self.D)), tf.reshape(evals_array.stack(), (batch_size,));
            
            @tf.function
            def comput(global_particle, global_eval, 
                       local_particle, local_eval,
                       particles, velocities,
                       boundary, norm_factors):
                for k in tf.range(maxiter):
                    #local_eval = tf.expand_dims(local_eval, axis = 1);
                    local_particle, local_eval = subloop();
                    print(local_particle, local_eval, global_eval)
                    #Tensor("StatefulPartitionedCall:0", shape=(?, 1, 5), dtype=float32)
                    #Tensor("StatefulPartitionedCall:1", shape=(?, 1), dtype=float32)
                    #Tensor("placeholder_4:0", shape=(64,), dtype=float32)
                    #local_eval = tf.squeeze(local_eval);
                    #argmin = tf.squeeze(tf.argmin(evals, 1, output_type = tf.int32));
                    #tmp = tf.range(batch_size);
                    #idx = tf.stack([tmp, argmin], axis=1);
                    
                    #local_particle, local_eval = tf.gather_nd(tf.transpose(particles, [0, 2, 1]), idx), tf.gather_nd(evals, idx);
                    
                    # 0.01
                    global_particle = tf.where(tf.less(global_eval, local_eval), 
                                               global_particle, local_particle);
                    global_eval = tf.where(tf.less(global_eval, local_eval), 
                                           global_eval, local_eval);
                    
                    local_diff = tf.expand_dims(local_particle, axis = 2) - particles;
                    global_diff = tf.expand_dims(global_particle, axis = 2) - particles;
                    
                    r1r2 = tf.random.uniform((2,), minval= 0, maxval= 1);
                    # 0.11 - 0.03 = 0.08
                    velocities = self.W * velocities + \
                                        self.C1 * r1r2[0] * local_diff + \
                                        self.C2 * r1r2[1] * global_diff;
                    
                    #norm_particles = (particles + velocities - tf.expand_dims(boundary[0], axis = 2)) / tf.expand_dims(norm_factors, axis = 2);
                    
                    #clipped_norm_particles = tf.clip_by_value(norm_particles, 0, 1);
                    #0.2 - 0.11 = 0.09
                    #particles = clipped_norm_particles * tf.expand_dims(norm_factors, axis = 2) + tf.expand_dims(boundary[0], axis = 2);
                    
                    # 0.13 - 0.11 = 0.02
                    #particles = particles + velocities;
                    particles = tf.add(particles, velocities);
                return global_particle, global_eval
        
        self.global_particle, self.global_eval = comput(global_particle, global_eval, 
                                              local_particle, local_eval,
                                              particles, velocities,
                                              boundary, norm_factors);
        return self.global_particle, self.global_eval
                                                        
    def build_batch_v3(self, y, maxiter, func, bounds, batch_size):
        with tf.variable_scope("swarm_intelligence"):
            maxiter = tf.constant(maxiter, dtype = tf.int32, name = 'PSO.maxiter');
            
            global_particle = tf.Variable(tf.zeros((batch_size, self.D,)), dtype = tf.float32, name = 'PSO.global_particle', trainable = False);
            global_eval = tf.Variable(tf.ones((batch_size,), dtype = tf.float32) * np.inf, dtype = tf.float32, name = 'PSO.global_eval', trainable = False);
            
            boundary = np.array(bounds, dtype = np.float32);
            boundary = tf.stack([tf.repeat(tf.expand_dims(boundary[0], axis = 0), batch_size, axis = 0), 
                                             tf.repeat(tf.expand_dims(boundary[1], axis = 0), batch_size, axis = 0)], name = 'PSO.boundary'); 
            norm_factors = boundary[1] - boundary[0];
            
            particles = tf.Variable(tf.random.uniform((batch_size, self.D, self.N, ), minval= 0, maxval= 1), 
                                    dtype = tf.float32, name = 'PSO.particles', trainable = False) * tf.expand_dims(norm_factors, axis = 2) + tf.expand_dims(boundary[0], axis = 2);
            velocities = tf.Variable(tf.zeros(shape = (batch_size, self.D, self.N, )), 
                                    dtype = tf.float32, name = 'PSO.velocities', trainable = False);
            
            # evals_array = tf.Variable(tf.ones((batch_size, 1)), dtype = tf.float32);
            # particles_array = tf.Variable(tf.zeros((batch_size, self.D,)), dtype = tf.float32)
                   
            @tf.function                 
            def subloop(particles):
                def inner_recursion_(j, limit, particles, y, func, evals_array, particles_array):
                    if j >= limit:
                        ff = evals_array.stack();
                        gg = particles_array.stack();
                        return ff, gg
                        #return evals_array, particles_array;
                    else:
                        tmp_p = tf.gather(particles, j);
                        tmp_y = tf.gather(y, j);
                        #ev = func(tf.transpose(tmp_p), 
                                   #tf.repeat(tf.expand_dims(tf.gather(y, j), axis = 0), self.N, axis = 0));
                                   
                        ev = func(tf.transpose(tmp_p), 
                                   tf.repeat(tf.reshape(tmp_y, shape = (1, -1)), self.N, axis = 0));
                        argmin = tf.argmin(ev, output_type = tf.int32);
                        #0.20
                        #a = tf.tensor_scatter_nd_update(evals_array, [[j]], tf.gather(ev, [[argmin]]));
                        #b = tf.tensor_scatter_nd_update(particles_array, [[j]], tf.gather(tf.transpose(tmp_p, [1, 0]), [argmin]));
                        
                        #a = slice_assign(evals_array, tf.gather(ev, [[argmin]]), slice(j, j + 1));
                        #b = slice_assign(particles_array, tf.gather(tf.transpose(tmp_p, [1, 0]), [argmin]), slice(j, j + 1));

                        #0.13
                        a = evals_array.write(j, tf.gather(ev, argmin));
                        b = particles_array.write(j, tf.gather(tf.transpose(tmp_p, [1, 0]), argmin));
                        return inner_recursion_(j + 1, limit, particles, y, func, a, b);
                
                return inner_recursion_(0, batch_size, particles, y, func, 
                                        tf.TensorArray(dtype = tf.float32, size = batch_size, dynamic_size= False),
                                        tf.TensorArray(dtype = tf.float32, size = batch_size, dynamic_size= False, element_shape= (self.D,)))
                #return inner_recursion_(0, batch_size, particles, y, func, 
                                        #evals_array,
                                        #particles_array)                
            
            @tf.function
            def comput(global_particle, global_eval, 
                       particles, velocities,
                       boundary, norm_factors):
                
                for k in tf.range(maxiter):
                    local_eval, local_particle = subloop(particles);
                    
                    global_particle = tf.where(tf.less(global_eval, local_eval), global_particle, local_particle);
                    global_eval = tf.where(tf.less(global_eval, local_eval), global_eval, local_eval);
                    
                    local_diff = tf.reshape(local_particle, shape = (batch_size, self.D, 1)) - particles;
                    global_diff = tf.reshape(global_particle, shape = (batch_size, self.D, 1)) - particles;
                    
                    r1r2 = tf.random.uniform((2, tf.shape(local_diff)[0], 1, 1), minval = 0, maxval = 1);
                    # 0.11 - 0.03 = 0.08
                    velocities = self.W * velocities + \
                                        self.C1 * r1r2[0] * local_diff + \
                                        self.C2 * r1r2[1] * global_diff;
                    
                    # norm_particles = (particles + velocities - tf.expand_dims(boundary[0], axis = 2)) / tf.expand_dims(norm_factors, axis = 2);
                    
                    # clipped_norm_particles = tf.clip_by_value(norm_particles, 0, 1);
                    #0.2 - 0.11 = 0.09
                    # particles = clipped_norm_particles * tf.expand_dims(norm_factors, axis = 2) + tf.expand_dims(boundary[0], axis = 2);
                    
                    # 0.13 - 0.11 = 0.02
                    particles = particles + velocities;
                    
                return global_particle, global_eval;
        
        self.global_particle, self.global_eval = comput(global_particle, global_eval, 
                                                        particles, velocities,
                                                        boundary, norm_factors);
            
#    def minimize(self, sess, paras_dict = {}):
#        loop = sess.run([self.global_particle, self.global_eval], feed_dict = paras_dict);
##        loop = sess.run([self.loop], feed_dict = paras_dict)[0];
#        return loop
    def minimize(self, ops, sess, paras_dict = {}):
        loop = sess.run([ops], feed_dict = paras_dict);
        return loop
    
    