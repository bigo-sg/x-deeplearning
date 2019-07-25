# Copyright (C) 2016-2018 Alibaba Group Holding Limited
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
from datetime import datetime
import sys
import os
import xdl
import subprocess
from xdl.python import pybind
from xdl.python.io.data_io import DataIO
from xdl.python.io.data_sharding import DataSharding
import threading


threadLock = threading.Lock()
threads = []
#thread_num=10

#input_list=[]
class myThread (threading.Thread):
   def __init__(self, threadID, inputlist, outputlist, converter_file):
      threading.Thread.__init__(self)
      self.threadID = threadID
      self.input = inputlist
      self.output = outputlist
      self._converter_file = converter_file
   def run(self):
      print "Starting " + self.name+ " tid "+ str(self.threadID)
      # Get lock to synchronize threads
      own_outputlist=[]
      for i in range(len(self.input[self.threadID])):
         own_outputlist.append(file_parsing(self.input[self.threadID][i],self._converter_file,self.threadID))
      
      threadLock.acquire()
      for i in range(len(own_outputlist)):
         self.output.append(own_outputlist[i])
    
      threadLock.release()
     
def file_parsing(path,converter_file, tid):
     currentDirectory = os.getcwd()
     hdp_cmd ="hadoop fs -get {} {}".format(path, currentDirectory)
     dirs_splited = path.split('/')
     gz_filename = dirs_splited[len(dirs_splited)-1]
     rm_cmd = "rm -f {}/{}".format(currentDirectory,gz_filename)
     exec_cmd(rm_cmd)
     p=exec_cmd(hdp_cmd)
     if p==0:
        print "after hdp_cmd"+"  "+ str(tid)
     rm_cmd = "rm -f {}/{}".format(currentDirectory,gz_filename.strip(".gz"))
     gzip_cmd = "gzip -d {}/{}".format(currentDirectory,gz_filename)
     exec_cmd(rm_cmd)
     exec_cmd(gzip_cmd)

     local_path = "{}/{}".format(currentDirectory,gz_filename.strip(".gz"))

     script_cmd = "python {}/{} {}".format(currentDirectory,converter_file,local_path)
     p=exec_cmd(script_cmd)
     if p==0:
        print str(tid)+"finish task"
     return local_path+".txt"

def exec_cmd(exec_cmd1):
     p = subprocess.call(exec_cmd1,
                         stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT,
                         shell=True)
     return p




class DataReader(DataIO):
    def __init__(self, ds_name, file_type=pybind.parsers.txt,
                 fs_type = None,
                 namenode="",
                 paths=None,
                 meta=None,
                 enable_state=True,
                 converter_file="main_newData.py", converter_threads=10):
        self._ds_name = ds_name
        self._paths = list()
        self._meta = meta
        self._fs_type = fs_type
        self._namenode = namenode
        self._converter_file = converter_file
        self.gz_hdfs_dir=""
        if paths is not None:
            assert isinstance(paths, list), "paths must be a list"

            for path in paths:
                fs_type, namenode, rpath = self._decode_path(path)

                if self._fs_type is not None and fs_type is not None:
                    assert fs_type == self._fs_type, "support only one filesystem %s"%self._fs_type
                else:
                    self._fs_type = fs_type

                if self._namenode != "" and namenode != "":
                    assert namenode == self._namenode, "support only one namenode %s"%self._namenode
                else:
                    self._namenode = namenode

                if rpath is not None:
                    self._paths.append(rpath)

        if self._fs_type is None:
            self._fs_type = pybind.fs.local

        super(DataReader, self).__init__(ds_name, file_type=file_type,
                                         fs_type=self._fs_type, namenode=self._namenode,
                                         enable_state=enable_state)

        # add path after failover
        self._sharding = DataSharding(self.fs())
        self._sharding.add_path(self._paths)

        shard_paths = self._sharding.partition(
            rank=xdl.get_task_index(), size=xdl.get_task_num())
        converted_path=[]
        thread_input= []
        inputlist=[]
        if self.gz_hdfs_dir!="":
          curr_thread=0
          thread_input.append([])
        else:
          curr_thread=-1
        threads_list=[]
        print "start"+"  "+str(datetime.now())
       
        for raw_path in shard_paths:
          arr = raw_path.split('/')
          filename = arr[-1]
          
          if filename.startswith('gz_hdfs_'):
             gz_hdfs_addr = self.gz_hdfs_dir+(filename.strip('gz_hdfs_')).strip('.txt')+'.gz'
             if len(thread_input[curr_thread])< ((len(shard_paths)/converter_threads)):
               thread_input[curr_thread].append(gz_hdfs_addr)
               #inputlist.append(gz_hdfs_addr)               
             else:
               if curr_thread < ((len(shard_paths)%converter_threads)): 
                  thread_input[curr_thread].append(gz_hdfs_addr)
               curr_thread = curr_thread+1
               thread_input.append([])
               #inputlist=[]
               if curr_thread >= ((len(shard_paths)%converter_threads)): 
                  thread_input[curr_thread].append(gz_hdfs_addr)
               #curr_thread = curr_thread+1
               print curr_thread       
             print gz_hdfs_addr
          else:
             converted_path.append(raw_path)
        
        if curr_thread>=0:
          #thread_input.append(inputlist)
          for i in range(curr_thread):
            threads_list.append(myThread(i, thread_input, converted_path,self._converter_file))
                
            threads_list[i].start()
          for i in range(curr_thread):
            threads_list[i].join()

          
        print "end"+"  "+str(datetime.now())      
        print('data paths:', converted_path)
        self.add_path(converted_path)
        if self._meta is not None:
            self.set_meta(self._meta)
    #def _exec_cmd(self, exec_cmd):
    #    p = subprocess.call(exec_cmd,
    #                     stdout=subprocess.PIPE,
    #                     stderr=subprocess.STDOUT,
    #                     shell=True)
    #    return p
    def _decode_path(self, path):
        '''
        hdfs://namenode/path
        kafka://namenode
        '''
        namenode = ""
        fs_type = None
        fpath = None
        if path.startswith('hdfs://'):
            fs_type = pybind.fs.hdfs
            arr = path.split('/', 3)
            assert len(arr) == 4
            namenode = arr[2]
            fpath = '/'+arr[3]
            if arr[3].endswith(".gz"):
                arr2 = path.split('/')
                if self.gz_hdfs_dir =="":
                   self.gz_hdfs_dir=path.strip(arr2[-1])
                if self.gz_hdfs_dir!= path.strip(arr2[-1]):
                  print "error file dir"
                  return
                currentDirectory = os.getcwd()
                
                #print "after hdp_cmd"
                dirs_splited = path.split('/')
                gz_filename = dirs_splited[len(dirs_splited)-1]
                local_path = "{}/gz_hdfs_{}".format(currentDirectory,gz_filename.strip(".gz"))
                xdl_file = open(local_path+".txt", "w")
                xdl_file.close()
                
                fpath =  local_path+".txt"
                fs_type = pybind.fs.local
                
        elif path.startswith('kafka://'):
            fs_type = pybind.fs.kafka
            arr = path.split('/', 2)
            assert len(arr) == 3
            namenode = arr[2]
        else:
            assert '://' not in path, "Unsupported path: %s" % path
            fpath = path

        return fs_type, namenode, fpath

