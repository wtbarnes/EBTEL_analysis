#ebtel_run.py

#Will Barnes
#7 May 2015

#Import needed modules
import os
import sys
import subprocess
import multiprocessing

def worker(func,*args):
    func(*args)

class Runner(object):
    
    def __init__(self,exec_directory,config_directory,**kwargs):
        self.exec_directory = exec_directory
        self.config_directory = config_directory
        
            
    def run_ebtel_single(self,config_file,**kwargs):
        if 'quiet' in kwargs and kwargs['quiet'] is True:
            quiet_option = ' quiet'
        else:
            quiet_option = ''
            
        output = subprocess.check_output([self.exec_directory+'ebtel-2fl ' + self.config_directory + config_file + quiet_option])
                
        if 'verbose' in kwargs and kwargs['verbose'] is True:
            print(output.decode(sys.stdout.encoding))
        
        
    def run_ebtel_multi_serial(self,**kwargs):
        if 'sub_dir' not in kwargs:
            kwargs['sub_dir'] = ''
        for name in os.listdir(self.config_directory+kwargs['sub_dir']):
            if os.path.isfile(self.config_directory+kwargs['sub_dir']+name):
                output = subprocess.Popen([(self.exec_directory+'ebtel-2fl'),(self.config_directory+kwargs['sub_dir']+name+' quiet')],stdout=subprocess.PIPE)
                print(output.communicate()[0].decode('ascii'))
                
                
    #This function deprecated--use not recommended            
    def run_ebtel_multi_parallel(self,**kwargs):
        if 'sub_dir' not in kwargs:
            kwargs['sub_dir'] = ''
        config_list = []
        for name in os.listdir(self.config_directory+kwargs['sub_dir']):
            if os.path.isfile(self.config_directory+kwargs['sub_dir']+name):
                config_list.append(kwargs['sub_dir']+name)
                
        pool = multiprocessing.Pool(processes=2)
        pool.map(self.run_ebtel_single,config_list)