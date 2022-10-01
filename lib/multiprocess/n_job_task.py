# -*- coding: utf-8 -*-
"""
Created on Sun Apr 26 11:25:05 2020

@author: bb
"""
import multiprocessing;

MAX_THREADS = int(multiprocessing.cpu_count() * 0.8);


class JobPool(object):
    def __init__(self, jobs, n_job=-1, verbose=0):
        self.jobs = jobs
        self.n_job = n_job if n_job != -1 else (MAX_THREADS if len(jobs) > MAX_THREADS else len(jobs));
        self.results_ = [];
        self.verbose = verbose;

    def start(self):
        from joblib import Parallel, delayed;
        out = Parallel(n_jobs=self.n_job,
                       verbose=self.verbose)(delayed(job.run)() for job in self.jobs);
        self.results_ = out;


class JobPoolForUnSerialize():
    #    not working
    def __init__(self, n_job=-1):
        self.results_ = [];
        self.n_job = n_job if n_job != -1 else (MAX_THREADS if len(jobs) > MAX_THREADS else len(jobs));

    def update_result(self, value):
        self.results_.append(value);

    def start(self, funcs, args):
        pool = multiprocessing.Pool(processes=self.n_job);
        #        results = [];
        if args is None:
            for f in funcs:
                pool.apply_async(f, callback=self.update_result);
        #                results.append(result);
        else:
            for f, arg in zip(funcs, args):
                pool.apply_async(f, arg, callback=self.update_result);
        #                f(*arg)
        #                results.append(result);
        pool.close();
        pool.join();


class JobPoolByCMD(object):
    def __init__(self):
        self.jobs = [];

    def join(self, title, key, system_tag):
        if system_tag == 'Windows':
            arg = "start /wait cmd /c \"title window for %s&& call activate trfl&& python %s\"" % (title, key);
        else:
            arg = "gnome-terminal --wait --title='%s' -e 'sh -c python %s'" % (title, key);
        self.jobs.append((arg));

    def start(self, n_processes=3, pool=None):
        import os;
        create_pool = pool is None;
        if create_pool:
            pool = multiprocessing.Pool(processes=n_processes);
        for arg in self.jobs:
            pool.apply_async(os.system, (arg,));

        pool.close();
        pool.join();


class abstractJob(object):
    def run(self):
        raise NotImplementedError('Job isnot implemented');


class exampleJob(abstractJob):
    def __init__(self, name):
        self.name = name;

    def run(self, ii=None):
        import time;
        import numpy as np;
        time.sleep(np.random.rand())
        return self.name, 'completed', {'name': self.name}, ii;


if __name__ == '__main__':
    nameList = ["One", "Two", "Three", "Four", "Five"];
    jobs = [exampleJob(name) for name in nameList];
    jobPool = JobPool(jobs, verbose=1);
    jobPool.start();
    print(jobPool.results_);
#    jobPool = JobPoolForUnSerialize();
#    jobPool.start([j.run for j in jobs], args = [(1,) for j in jobs] );
#    print(jobPool.results_);
